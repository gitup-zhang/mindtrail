import hashlib
import json
import math
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Set

import numpy as np

from ..core.config import env
from ..core.constants import SECTOR_CONFIGS
from ..core.db import db, q
from ..core.vector_store import vector_store as store
from ..ops.dynamics import (
    applyRetrievalTraceReinforcementToMemory,
    calculateCrossSectorResonanceScore,
    propagateAssociativeReinforcementToLinkedNodes,
)
from ..utils.chunking import chunk_text
from ..utils.keyword import compute_keyword_overlap, extract_keywords
from ..utils.text import canonical_token_set, stable_text_fallback_hash
from ..utils.vectors import buf_to_vec, cos_sim, vec_to_buf
from .decay import calc_recency_score as calc_recency_score_decay
from .decay import dec_q, inc_q, on_query_hit
from .embed import calc_mean_vec, embed_for_sector, embed_multi_sector

SCORING_WEIGHTS = {
    "similarity": 0.35,
    "overlap": 0.20,
    "waypoint": 0.15,
    "recency": 0.10,
    "tag_match": 0.20,
}

HYBRID_PARAMS = {
    "tau": 3.0,
    "gamma": 0.2,
}

SECTOR_RELATIONSHIPS = {
    "semantic": {"procedural": 0.8, "episodic": 0.6, "reflective": 0.7, "emotional": 0.4},
    "procedural": {"semantic": 0.8, "episodic": 0.6, "reflective": 0.6, "emotional": 0.3},
    "episodic": {"reflective": 0.8, "semantic": 0.6, "procedural": 0.6, "emotional": 0.7},
    "reflective": {"episodic": 0.8, "semantic": 0.7, "procedural": 0.6, "emotional": 0.6},
    "emotional": {"episodic": 0.7, "reflective": 0.6, "semantic": 0.4, "procedural": 0.3},
}


async def embed_query_for_all_sectors(query: str, sectors: List[str]) -> Dict[str, List[float]]:
    return {sector: await embed_for_sector(query, sector) for sector in sectors}


async def compute_tag_match_score(memory_id: str, query_tokens: Set[str]) -> float:
    row = q.get_mem(memory_id)
    if not row or not row["tags"]:
        return 0.0

    try:
        tags = json.loads(row["tags"])
    except json.JSONDecodeError:
        return 0.0

    if not isinstance(tags, list) or not tags:
        return 0.0

    matches = 0
    for tag in tags:
        lowered = str(tag).lower()
        if lowered in query_tokens:
            matches += 2
        elif any(lowered in token or token in lowered for token in query_tokens):
            matches += 1
    return min(1.0, matches / max(1, len(tags) * 2))


def compress_vec_for_storage(vector: List[float], target_dim: int = 128) -> List[float]:
    if len(vector) <= target_dim:
        return vector

    compressed = [0.0] * target_dim
    bucket_size = len(vector) / target_dim
    for index in range(target_dim):
        start = int(index * bucket_size)
        end = int((index + 1) * bucket_size)
        bucket = vector[start:end]
        compressed[index] = sum(bucket) / len(bucket) if bucket else 0.0

    norm = math.sqrt(sum(value * value for value in compressed))
    if norm > 0:
        compressed = [value / norm for value in compressed]
    return compressed


def classify_content(content: str, metadata: Any = None) -> Dict[str, Any]:
    sector_override = metadata.get("sector") if isinstance(metadata, dict) else None
    if sector_override in SECTOR_CONFIGS:
        return {"primary": sector_override, "additional": [], "confidence": 1.0}

    scores = {sector: 0.0 for sector in SECTOR_CONFIGS}
    for sector, config in SECTOR_CONFIGS.items():
        score = 0.0
        for pattern in config["patterns"]:
            score += len(pattern.findall(content)) * config["weight"]
        scores[sector] = score

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    primary, primary_score = ranked[0]
    threshold = max(1.0, primary_score * 0.3)
    additional = [sector for sector, score in ranked[1:] if score > 0 and score >= threshold]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    confidence = min(1.0, primary_score / (primary_score + second_score + 1)) if primary_score > 0 else 0.2

    return {
        "primary": primary if primary_score > 0 else "semantic",
        "additional": additional,
        "confidence": confidence,
    }


def calc_decay(sector: str, initial_salience: float, days_since: float) -> float:
    config = SECTOR_CONFIGS.get(sector)
    if not config:
        return initial_salience
    return max(0.0, min(1.0, initial_salience * math.exp(-config["decay_lambda"] * days_since)))


def boosted_sim(similarity: float) -> float:
    return 1 - math.exp(-HYBRID_PARAMS["tau"] * similarity)


def compute_simhash(text: str) -> str:
    tokens = canonical_token_set(text)
    if not tokens:
        return stable_text_fallback_hash(text)

    accumulator = [0] * 64
    for token in tokens:
        digest = int(hashlib.blake2b(token.encode("utf-8"), digest_size=8).hexdigest(), 16)
        for bit in range(64):
            accumulator[bit] += 1 if digest & (1 << bit) else -1

    out = 0
    for bit, weight in enumerate(accumulator):
        if weight > 0:
            out |= 1 << bit
    return f"{out:016x}"


def hamming_dist(first: str, second: str) -> int:
    return (int(first, 16) ^ int(second, 16)).bit_count()


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def extract_essence(raw_text: str, sector: str, max_len: int) -> str:
    del sector
    if not env.use_summary_only or len(raw_text) <= max_len:
        return raw_text

    sentences = [
        item.strip()
        for item in re.split(r"(?<=[.!?。！？；;])\s*", raw_text)
        if len(item.strip()) > 10
    ]
    if not sentences:
        return raw_text[:max_len]

    ranked: List[Dict[str, Any]] = []
    for index, sentence in enumerate(sentences):
        score = 0
        if index == 0:
            score += 10
        if re.search(r"\b\d{4}-\d{2}-\d{2}\b", sentence):
            score += 6
        if re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+", sentence):
            score += 3
        if len(sentence) < 100:
            score += 2
        ranked.append({"text": sentence, "score": score, "index": index})

    ranked.sort(key=lambda item: item["score"], reverse=True)
    selected: List[Dict[str, Any]] = []
    current_length = 0
    for item in ranked:
        if current_length + len(item["text"]) + 1 > max_len:
            continue
        selected.append(item)
        current_length += len(item["text"]) + 1

    selected.sort(key=lambda item: item["index"])
    return " ".join(item["text"] for item in selected)


def compute_token_overlap(query_tokens: Set[str], memory_tokens: Set[str]) -> float:
    if not query_tokens:
        return 0.0
    return len(query_tokens.intersection(memory_tokens)) / len(query_tokens)


def compute_hybrid_score(
    similarity: float,
    token_overlap: float,
    waypoint_weight: float,
    recency: float,
    keyword_bonus: float = 0.0,
    tag_match: float = 0.0,
) -> float:
    raw = (
        SCORING_WEIGHTS["similarity"] * boosted_sim(similarity)
        + SCORING_WEIGHTS["overlap"] * token_overlap
        + SCORING_WEIGHTS["waypoint"] * waypoint_weight
        + SCORING_WEIGHTS["recency"] * recency
        + SCORING_WEIGHTS["tag_match"] * tag_match
        + keyword_bonus
    )
    return sigmoid(raw)


async def create_single_waypoint(memory_id: str, mean_vector: List[float], user_id: Optional[str] = None) -> None:
    rows = q.all_mem_by_user(user_id, 1000, 0) if user_id else q.all_mem(1000, 0)
    current = np.array(mean_vector, dtype=np.float32)
    best_id: Optional[str] = None
    best_similarity = -1.0

    for row in rows:
        if row["id"] == memory_id or not row["mean_vec"]:
            continue
        candidate = np.array(buf_to_vec(row["mean_vec"]), dtype=np.float32)
        similarity = cos_sim(current, candidate)
        if similarity > best_similarity:
            best_similarity = similarity
            best_id = row["id"]

    now = int(time.time() * 1000)
    db.execute(
        """
        INSERT OR REPLACE INTO waypoints (src_id, dst_id, user_id, weight, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            memory_id,
            best_id or memory_id,
            user_id or "anonymous",
            float(best_similarity) if best_id else 1.0,
            now,
            now,
        ),
    )
    db.commit()


async def create_inter_memory_waypoints(
    memory_id: str,
    sector: str,
    vector: List[float],
    user_id: Optional[str] = None,
) -> None:
    neighbours = await store.search(vector, sector, 8, {"user_id": user_id})
    now = int(time.time() * 1000)
    for neighbour in neighbours:
        target_id = neighbour["id"]
        similarity = float(neighbour["similarity"])
        if target_id == memory_id or similarity < 0.75:
            continue
        for src_id, dst_id in ((memory_id, target_id), (target_id, memory_id)):
            db.execute(
                """
                INSERT OR REPLACE INTO waypoints (src_id, dst_id, user_id, weight, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (src_id, dst_id, user_id or "anonymous", min(1.0, similarity), now, now),
            )
    db.commit()


async def calc_multi_vec_fusion_score(
    memory_id: str,
    query_embeddings: Dict[str, List[float]],
    weights: Dict[str, float],
) -> float:
    vectors = await store.getVectorsById(memory_id)
    total = 0.0
    total_weight = 0.0
    weight_map = {
        "semantic": weights.get("semantic_dimension_weight", 0.8),
        "emotional": weights.get("emotional_dimension_weight", 0.6),
        "procedural": weights.get("procedural_dimension_weight", 0.7),
        "episodic": weights.get("temporal_dimension_weight", 0.7),
        "reflective": weights.get("reflective_dimension_weight", 0.5),
    }

    for vector in vectors:
        query_vector = query_embeddings.get(vector.sector)
        if not query_vector:
            continue
        sector_weight = weight_map.get(vector.sector, 0.5)
        total += cos_sim(vector.vector, query_vector) * sector_weight
        total_weight += sector_weight

    return total / total_weight if total_weight > 0 else 0.0


async def add_hsg_memory(
    content: str,
    tags: Optional[str] = None,
    metadata: Any = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    simhash = compute_simhash(content)
    existing = db.fetchone(
        "SELECT * FROM memories WHERE simhash=? ORDER BY salience DESC LIMIT 1",
        (simhash,),
    )
    if existing and hamming_dist(simhash, existing["simhash"]) <= 3:
        now = int(time.time() * 1000)
        boosted = min(1.0, float(existing["salience"] or 0.0) + 0.15)
        db.execute(
            "UPDATE memories SET last_seen_at=?, salience=?, updated_at=? WHERE id=?",
            (now, boosted, now, existing["id"]),
        )
        db.commit()
        return {
            "id": existing["id"],
            "primary_sector": existing["primary_sector"],
            "sectors": [existing["primary_sector"]],
            "deduplicated": True,
        }

    memory_id = str(uuid.uuid4())
    now = int(time.time() * 1000)
    chunks = chunk_text(content)
    classification = classify_content(content, metadata)
    sectors = [classification["primary"], *classification["additional"]]
    max_segment_row = db.fetchone("SELECT COALESCE(MAX(segment), 0) AS max_seg FROM memories")
    current_segment = max_segment_row["max_seg"] if max_segment_row else 0
    count_row = db.fetchone("SELECT COUNT(*) AS count FROM memories WHERE segment=?", (current_segment,))
    if count_row and count_row["count"] >= env.seg_size:
        current_segment += 1

    q.ins_mem(
        id=memory_id,
        user_id=user_id or "anonymous",
        segment=current_segment,
        content=extract_essence(content, classification["primary"], env.summary_max_length),
        simhash=simhash,
        primary_sector=classification["primary"],
        tags=tags,
        meta=json.dumps(metadata or {}),
        created_at=now,
        updated_at=now,
        last_seen_at=now,
        salience=max(0.0, min(1.0, 0.4 + 0.1 * len(classification["additional"]))),
        decay_lambda=SECTOR_CONFIGS[classification["primary"]]["decay_lambda"],
        version=1,
        feedback_score=0,
    )

    embedding_results = await embed_multi_sector(memory_id, content, sectors, chunks if len(chunks) > 1 else None)
    for item in embedding_results:
        await store.storeVector(memory_id, item["sector"], item["vector"], item["dim"], user_id or "anonymous")

    mean_vector = calc_mean_vec(embedding_results, sectors)
    db.execute("UPDATE memories SET mean_dim=?, mean_vec=? WHERE id=?", (len(mean_vector), vec_to_buf(mean_vector), memory_id))
    if len(mean_vector) > 128:
        db.execute("UPDATE memories SET compressed_vec=? WHERE id=?", (vec_to_buf(compress_vec_for_storage(mean_vector)), memory_id))

    await create_single_waypoint(memory_id, mean_vector, user_id)
    primary_vector = next(item["vector"] for item in embedding_results if item["sector"] == classification["primary"])
    await create_inter_memory_waypoints(memory_id, classification["primary"], primary_vector, user_id)
    db.commit()

    return {
        "id": memory_id,
        "content": content,
        "primary_sector": classification["primary"],
        "sectors": sectors,
        "chunks": len(chunks),
    }


async def expand_via_waypoints(memory_ids: List[str], max_expand: int = 10) -> List[Dict[str, Any]]:
    expanded: List[Dict[str, Any]] = []
    visited = set(memory_ids)
    queue = [{"id": memory_id, "weight": 1.0, "path": [memory_id]} for memory_id in memory_ids]

    while queue and len(expanded) < max_expand:
        current = queue.pop(0)
        neighbours = db.fetchall(
            "SELECT dst_id, weight FROM waypoints WHERE src_id=? ORDER BY weight DESC",
            (current["id"],),
        )
        for row in neighbours:
            target_id = row["dst_id"]
            if target_id in visited:
                continue
            weight = min(1.0, max(0.0, float(row["weight"] or 0.0)))
            path_weight = current["weight"] * weight * 0.8
            if path_weight < 0.1:
                continue
            item = {"id": target_id, "weight": path_weight, "path": current["path"] + [target_id]}
            expanded.append(item)
            visited.add(target_id)
            queue.append(item)
            if len(expanded) >= max_expand:
                break

    return expanded


async def hsg_query(query_text: str, limit: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    filters = filters or {}
    inc_q()
    try:
        query_classification = classify_content(query_text)
        query_tokens = canonical_token_set(query_text)
        query_keywords = extract_keywords(query_text, env.keyword_min_length)
        sectors = filters.get("sectors") or list(SECTOR_CONFIGS.keys())
        query_embeddings = await embed_query_for_all_sectors(query_text, sectors)
        weights = {
            "semantic_dimension_weight": 1.2 if query_classification["primary"] == "semantic" else 0.8,
            "emotional_dimension_weight": 1.5 if query_classification["primary"] == "emotional" else 0.6,
            "procedural_dimension_weight": 1.3 if query_classification["primary"] == "procedural" else 0.7,
            "temporal_dimension_weight": 1.4 if query_classification["primary"] == "episodic" else 0.7,
            "reflective_dimension_weight": 1.1 if query_classification["primary"] == "reflective" else 0.5,
        }

        search_results_by_sector: Dict[str, List[Dict[str, Any]]] = {}
        all_similarities: List[float] = []
        candidate_ids: Set[str] = set()
        for sector in sectors:
            results = await store.search(query_embeddings[sector], sector, limit * 3, {"user_id": filters.get("user_id")})
            search_results_by_sector[sector] = results
            for item in results:
                all_similarities.append(item["similarity"])
                candidate_ids.add(item["id"])

        expanded = []
        average_similarity = sum(all_similarities) / len(all_similarities) if all_similarities else 0.0
        if average_similarity < 0.55 and candidate_ids:
            expanded = await expand_via_waypoints(list(candidate_ids), limit * 2)
            candidate_ids.update(item["id"] for item in expanded)

        results: List[Dict[str, Any]] = []
        for memory_id in candidate_ids:
            row = q.get_mem(memory_id)
            if not row:
                continue

            memory = dict(row)
            if filters.get("min_salience") and memory["salience"] < filters["min_salience"]:
                continue
            if filters.get("user_id") and memory["user_id"] != filters["user_id"]:
                continue

            fusion_score = await calc_multi_vec_fusion_score(memory_id, query_embeddings, weights)
            best_similarity = await calculateCrossSectorResonanceScore(memory["primary_sector"], query_classification["primary"], fusion_score)
            for sector_results in search_results_by_sector.values():
                for item in sector_results:
                    if item["id"] == memory_id:
                        best_similarity = max(best_similarity, item["similarity"])

            penalty = 1.0
            if memory["primary_sector"] != query_classification["primary"]:
                penalty = SECTOR_RELATIONSHIPS.get(query_classification["primary"], {}).get(memory["primary_sector"], 0.3)
            adjusted_similarity = best_similarity * penalty
            expansion = next((item for item in expanded if item["id"] == memory_id), None)
            waypoint_weight = min(1.0, max(0.0, expansion["weight"] if expansion else 0.0))
            keyword_score = compute_keyword_overlap(query_keywords, extract_keywords(memory["content"], env.keyword_min_length)) * 0.15
            tag_match = await compute_tag_match_score(memory_id, query_tokens)
            token_overlap = compute_token_overlap(query_tokens, canonical_token_set(memory["content"]))
            recency_score = calc_recency_score_decay(memory["last_seen_at"])
            score = compute_hybrid_score(adjusted_similarity, token_overlap, waypoint_weight, recency_score, keyword_score, tag_match)

            result = {
                "id": memory_id,
                "content": memory["content"],
                "score": score,
                "primary_sector": memory["primary_sector"],
                "path": expansion["path"] if expansion else [memory_id],
                "salience": calc_decay(memory["primary_sector"], memory["salience"], (time.time() * 1000 - memory["last_seen_at"]) / 86_400_000.0),
                "last_seen_at": memory["last_seen_at"],
                "tags": json.loads(memory["tags"] or "[]"),
                "metadata": json.loads(memory["meta"] or "{}"),
            }
            if filters.get("debug"):
                result["_debug"] = {
                    "adjusted_similarity": adjusted_similarity,
                    "token_overlap": token_overlap,
                    "recency": recency_score,
                    "waypoint": waypoint_weight,
                    "tag_match": tag_match,
                    "sector_penalty": penalty,
                }
            results.append(result)

        results.sort(key=lambda item: item["score"], reverse=True)
        top_results = results[:limit]
        for item in top_results:
            reinforced_salience = await applyRetrievalTraceReinforcementToMemory(item["id"], item["salience"])
            now = int(time.time() * 1000)
            db.execute("UPDATE memories SET salience=?, last_seen_at=?, updated_at=? WHERE id=?", (reinforced_salience, now, now, item["id"]))

            if len(item["path"]) > 1:
                waypoint_rows = db.fetchall("SELECT dst_id, weight FROM waypoints WHERE src_id=?", (item["id"],))
                updates = await propagateAssociativeReinforcementToLinkedNodes(
                    item["id"],
                    reinforced_salience,
                    [{"target_id": row["dst_id"], "weight": row["weight"]} for row in waypoint_rows],
                )
                for update in updates:
                    linked = q.get_mem(update["node_id"])
                    if not linked:
                        continue
                    new_salience = max(0.0, min(1.0, float(linked["salience"] or 0.0) + HYBRID_PARAMS["gamma"] * (reinforced_salience - float(linked["salience"] or 0.0))))
                    db.execute("UPDATE memories SET salience=?, last_seen_at=?, updated_at=? WHERE id=?", (new_salience, now, now, update["node_id"]))

            await on_query_hit(item["id"], item["primary_sector"], lambda text: embed_for_sector(text, item["primary_sector"]))

        db.commit()
        return top_results
    finally:
        dec_q()
