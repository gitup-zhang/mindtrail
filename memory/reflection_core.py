import json
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

from ..core.db import db, q
from ..utils.text import canonical_token_set, contains_cjk
from ..utils.vectors import buf_to_vec, cos_sim
from .engine import add_hsg_memory


def _safe_meta(raw: Optional[str]) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _memory_tokens(memory: Dict[str, Any]) -> set[str]:
    return canonical_token_set(memory["content"])


def _token_similarity(first: Dict[str, Any], second: Dict[str, Any]) -> float:
    left = _memory_tokens(first)
    right = _memory_tokens(second)
    if not left or not right:
        return 0.0
    union = left.union(right)
    if not union:
        return 0.0
    return len(left.intersection(right)) / len(union)


def _vector_similarity(first: Dict[str, Any], second: Dict[str, Any]) -> float:
    if not first.get("mean_vec") or not second.get("mean_vec"):
        return 0.0
    return max(
        0.0,
        cos_sim(buf_to_vec(first["mean_vec"]), buf_to_vec(second["mean_vec"])),
    )


def _memory_similarity(first: Dict[str, Any], second: Dict[str, Any]) -> float:
    token_score = _token_similarity(first, second)
    vector_score = _vector_similarity(first, second)
    if token_score == 0.0:
        return vector_score
    if vector_score == 0.0:
        return token_score
    return 0.55 * vector_score + 0.45 * token_score


def _is_reflection_candidate(memory: Dict[str, Any]) -> bool:
    if memory["primary_sector"] == "reflective":
        return False

    meta = _safe_meta(memory.get("meta"))
    if meta.get("is_root") or meta.get("is_child"):
        return False
    if meta.get("consolidated_in"):
        return False
    return True


def _cluster_memories(
    memories: List[Dict[str, Any]],
    min_cluster_size: int,
    similarity_threshold: float,
) -> List[List[Dict[str, Any]]]:
    grouped: Dict[tuple[Optional[str], str], List[Dict[str, Any]]] = defaultdict(list)
    for memory in memories:
        if not _is_reflection_candidate(memory):
            continue
        grouped[(memory.get("user_id"), memory["primary_sector"])].append(memory)

    clusters: List[List[Dict[str, Any]]] = []
    for group in grouped.values():
        group.sort(
            key=lambda memory: (
                float(memory.get("salience") or 0.0),
                int(memory.get("last_seen_at") or memory.get("created_at") or 0),
            ),
            reverse=True,
        )
        used: set[str] = set()
        for seed in group:
            if seed["id"] in used:
                continue

            cluster = [seed]
            used.add(seed["id"])
            changed = True
            while changed:
                changed = False
                for candidate in group:
                    if candidate["id"] in used:
                        continue
                    if any(
                        _memory_similarity(candidate, existing) >= similarity_threshold
                        for existing in cluster
                    ):
                        cluster.append(candidate)
                        used.add(candidate["id"])
                        changed = True

            if len(cluster) >= min_cluster_size:
                clusters.append(cluster)

    clusters.sort(key=lambda cluster: (len(cluster), sum(float(item.get("salience") or 0.0) for item in cluster)), reverse=True)
    return clusters


def _shared_themes(cluster: List[Dict[str, Any]], limit: int = 4) -> List[str]:
    counts: Counter[str] = Counter()
    for memory in cluster:
        counts.update(_memory_tokens(memory))

    minimum_support = max(2, min(len(cluster), 3))
    common = [
        token
        for token, count in counts.most_common()
        if len(token) >= 3 and count >= minimum_support
    ]
    return common[:limit]


def _snippet(text: str, max_len: int = 140) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_len:
        return normalized
    return normalized[: max_len - 3].rstrip() + "..."


def _build_reflection_text(cluster: List[Dict[str, Any]]) -> str:
    sector = cluster[0]["primary_sector"]
    themes = _shared_themes(cluster)
    chinese = sum(1 for memory in cluster if contains_cjk(memory["content"])) >= max(1, len(cluster) // 2)
    if chinese:
        sector_names = {
            "semantic": "语义",
            "episodic": "情节",
            "procedural": "程序",
            "emotional": "情绪",
            "reflective": "反思",
        }
        header = f"这是对 {len(cluster)} 条{sector_names.get(sector, sector)}记忆的反思总结。"
        theme_line = f"共同主题：{'、'.join(themes)}。" if themes else "共同主题：这些记忆反复指向同一类信息。"
        lines = [header, theme_line, "代表性记忆："]
    else:
        header = f"Reflection over {len(cluster)} {sector} memories."
        theme_line = f"Shared themes: {', '.join(themes)}." if themes else "Shared themes: recurring related details."
        lines = [header, theme_line, "Representative memories:"]
    for memory in cluster[:3]:
        lines.append(f"- {_snippet(memory['content'])}")
    if len(cluster) > 3:
        if chinese:
            lines.append(f"- 以及另外 {len(cluster) - 3} 条相关记忆")
        else:
            lines.append(f"- plus {len(cluster) - 3} additional related memories")
    return "\n".join(lines)


def _reflection_salience(cluster: List[Dict[str, Any]]) -> float:
    avg_salience = sum(float(memory.get("salience") or 0.0) for memory in cluster) / max(1, len(cluster))
    return min(1.0, 0.45 + 0.08 * min(len(cluster), 5) + 0.35 * avg_salience)


async def _mark_cluster_consolidated(cluster: List[Dict[str, Any]], reflection_id: str) -> int:
    now = int(time.time() * 1000)
    updated = 0
    for memory in cluster:
        meta = _safe_meta(memory.get("meta"))
        meta["consolidated_in"] = reflection_id
        meta["consolidated_at"] = now
        meta["consolidated_kind"] = "auto_reflection"
        boosted_salience = min(1.0, float(memory.get("salience") or 0.0) + 0.05)
        db.execute(
            "UPDATE memories SET meta=?, salience=?, last_seen_at=?, updated_at=? WHERE id=?",
            (json.dumps(meta), boosted_salience, now, now, memory["id"]),
        )
        updated += 1
    return updated


async def run_reflection(
    user_id: Optional[str] = None,
    limit: int = 100,
    min_cluster_size: int = 2,
    similarity_threshold: float = 0.28,
) -> Dict[str, Any]:
    rows = q.all_mem_by_user(user_id, limit, 0) if user_id else q.all_mem(limit, 0)
    memories = [dict(row) for row in rows]
    eligible = [memory for memory in memories if _is_reflection_candidate(memory)]
    clusters = _cluster_memories(memories, min_cluster_size, similarity_threshold)

    created = 0
    consolidated = 0
    reflection_ids: List[str] = []
    now = int(time.time() * 1000)

    for cluster in clusters:
        summary = _build_reflection_text(cluster)
        source_ids = [memory["id"] for memory in cluster]
        metadata = {
            "sector": "reflective",
            "type": "auto_reflection",
            "generated_at": now,
            "source_ids": source_ids,
            "cluster_size": len(cluster),
            "source_sector": cluster[0]["primary_sector"],
        }
        tags = json.dumps(["reflect:auto", f"sector:{cluster[0]['primary_sector']}"])
        result = await add_hsg_memory(summary, tags=tags, metadata=metadata, user_id=user_id)
        reflection_id = result["id"]
        reflection_ids.append(reflection_id)
        db.execute(
            "UPDATE memories SET salience=?, updated_at=?, last_seen_at=? WHERE id=?",
            (_reflection_salience(cluster), now, now, reflection_id),
        )
        consolidated += await _mark_cluster_consolidated(cluster, reflection_id)
        created += 1

    db.commit()
    return {
        "processed": len(memories),
        "eligible": len(eligible),
        "clusters": len(clusters),
        "created": created,
        "consolidated": consolidated,
        "reflection_ids": reflection_ids,
        "user_id": user_id,
    }
