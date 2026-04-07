import json
import time
import uuid
from typing import Any, Dict, List, Optional

from ..core.db import db


async def insert_fact(
    subject: str,
    predicate: str,
    object_value: str,
    valid_from: Optional[int] = None,
    confidence: float = 1.0,
    metadata: Optional[Dict[str, Any]] = None,
    user_id: Optional[str] = None,
) -> str:
    fact_id = str(uuid.uuid4())
    now = int(time.time() * 1000)
    valid_from_ts = valid_from if valid_from is not None else now
    clauses = ["subject=?", "predicate=?", "valid_to IS NULL"]
    params: List[Any] = [subject, predicate]
    if user_id is None:
        clauses.append("user_id IS NULL")
    else:
        clauses.append("user_id=?")
        params.append(user_id)
    existing = db.fetchall(
        f"SELECT id, valid_from FROM temporal_facts WHERE {' AND '.join(clauses)}",
        tuple(params),
    )
    for row in existing:
        if row["valid_from"] < valid_from_ts:
            db.execute("UPDATE temporal_facts SET valid_to=?, last_updated=? WHERE id=?", (valid_from_ts - 1, now, row["id"]))

    db.execute(
        """
        INSERT INTO temporal_facts (id, user_id, subject, predicate, object, valid_from, valid_to, confidence, last_updated, metadata)
        VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)
        """,
        (fact_id, user_id, subject, predicate, object_value, valid_from_ts, confidence, now, json.dumps(metadata) if metadata else None),
    )
    db.commit()
    return fact_id


async def update_fact(fact_id: str, confidence: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
    updates = []
    params: List[Any] = []
    if confidence is not None:
        updates.append("confidence=?")
        params.append(confidence)
    if metadata is not None:
        updates.append("metadata=?")
        params.append(json.dumps(metadata))
    if not updates:
        return
    updates.append("last_updated=?")
    params.append(int(time.time() * 1000))
    params.append(fact_id)
    db.execute(f"UPDATE temporal_facts SET {', '.join(updates)} WHERE id=?", tuple(params))
    db.commit()


async def invalidate_fact(fact_id: str, valid_to: Optional[int] = None) -> None:
    now = int(time.time() * 1000)
    db.execute("UPDATE temporal_facts SET valid_to=?, last_updated=? WHERE id=?", (valid_to or now, now, fact_id))
    db.commit()


async def delete_fact(fact_id: str) -> None:
    db.execute("DELETE FROM temporal_facts WHERE id=?", (fact_id,))
    db.commit()


async def batch_insert_facts(facts: List[Dict[str, Any]]) -> List[str]:
    ids: List[str] = []
    for fact in facts:
        fact_id = await insert_fact(
            subject=fact["subject"],
            predicate=fact["predicate"],
            object_value=fact["object"],
            valid_from=fact.get("valid_from"),
            confidence=fact.get("confidence", 1.0),
            metadata=fact.get("metadata"),
            user_id=fact.get("user_id"),
        )
        ids.append(fact_id)
    return ids


async def apply_confidence_decay(decay_rate: float = 0.01) -> int:
    now = int(time.time() * 1000)
    db.execute(
        """
        UPDATE temporal_facts
        SET confidence = MAX(0.1, confidence * (1 - ? * ((? - valid_from) / 86400000.0))),
            last_updated = ?
        WHERE valid_to IS NULL AND confidence > 0.1
        """,
        (decay_rate, now, now),
    )
    db.commit()
    return db.conn.total_changes if db.conn else 0
