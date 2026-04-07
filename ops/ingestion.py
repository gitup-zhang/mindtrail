import json
import time
import uuid
from typing import Any, Dict, List, Optional

from ..core.config import env
from ..core.db import db, q
from ..memory.engine import add_hsg_memory
from .extractor import extract_text


def split_text(text: str, section_size: int) -> List[str]:
    if len(text) <= section_size:
        return [text]

    sections: List[str] = []
    current = ""
    for paragraph in text.split("\n\n"):
        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) > section_size and current:
            sections.append(current.strip())
            current = paragraph
        else:
            current = candidate

    if current.strip():
        sections.append(current.strip())
    return sections


async def make_root_memory(text: str, extraction: Dict[str, Any], tags_json: str, meta: Optional[Dict[str, Any]], user_id: Optional[str]) -> str:
    summary = text[:500] + "..." if len(text) > 500 else text
    section_count = int(len(text) / env.section_size_chars) + 1
    content_type = extraction["metadata"]["content_type"].upper()
    memory_id = str(uuid.uuid4())
    now = int(time.time() * 1000)
    merged_meta = dict(meta or {})
    merged_meta.update(extraction["metadata"])
    merged_meta.update({"is_root": True, "ingestion_strategy": "root-child", "ingested_at": now})

    q.ins_mem(
        id=memory_id,
        user_id=user_id or "anonymous",
        segment=0,
        content=f"[Document: {content_type}]\n\n{summary}\n\n[Split into {section_count} sections]",
        primary_sector="reflective",
        tags=tags_json,
        meta=json.dumps(merged_meta, default=str),
        created_at=now,
        updated_at=now,
        last_seen_at=now,
        salience=1.0,
        decay_lambda=0.1,
        feedback_score=0,
    )
    return memory_id


async def make_child_memory(
    text: str,
    index: int,
    total: int,
    root_id: str,
    tags_json: str,
    meta: Optional[Dict[str, Any]],
    user_id: Optional[str],
) -> str:
    child_meta = dict(meta or {})
    child_meta.update(
        {
            "is_child": True,
            "section_index": index,
            "total_sections": total,
            "parent_id": root_id,
        }
    )
    result = await add_hsg_memory(text, tags_json, child_meta, user_id)
    return result["id"]


async def link_root_to_child(root_id: str, child_id: str, user_id: Optional[str]) -> None:
    now = int(time.time() * 1000)
    db.execute(
        "INSERT INTO waypoints (src_id, dst_id, user_id, weight, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        (root_id, child_id, user_id or "anonymous", 1.0, now, now),
    )
    db.commit()


async def ingest_document(
    content_type: str,
    data: Any,
    meta: Optional[Dict[str, Any]] = None,
    cfg: Optional[Dict[str, Any]] = None,
    user_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    config = cfg or {}
    root_threshold = int(config.get("root_threshold", env.large_document_threshold))
    section_size = int(config.get("section_size", env.section_size_chars))
    tags_json = json.dumps(tags or [])

    extraction = await extract_text(content_type, data)
    text = extraction["text"]
    extraction_meta = extraction["metadata"]
    total_tokens = extraction_meta["estimated_tokens"]

    if total_tokens <= root_threshold:
        merged_meta = dict(meta or {})
        merged_meta.update(extraction_meta)
        merged_meta.update({"ingestion_strategy": "single", "ingested_at": int(time.time() * 1000)})
        result = await add_hsg_memory(text, tags_json, merged_meta, user_id)
        return {
            "root_memory_id": result["id"],
            "child_count": 0,
            "total_tokens": total_tokens,
            "strategy": "single",
            "extraction": extraction_meta,
        }

    sections = split_text(text, section_size)
    root_id = await make_root_memory(text, extraction, tags_json, meta, user_id)
    child_ids: List[str] = []
    for index, section in enumerate(sections):
        child_id = await make_child_memory(section, index, len(sections), root_id, tags_json, meta, user_id)
        child_ids.append(child_id)
        await link_root_to_child(root_id, child_id, user_id)

    return {
        "root_memory_id": root_id,
        "child_count": len(child_ids),
        "total_tokens": total_tokens,
        "strategy": "root-child",
        "extraction": extraction_meta,
    }
