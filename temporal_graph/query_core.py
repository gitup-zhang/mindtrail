import json
import time
from typing import Any, Dict, List, Optional

from ..core.db import db


def format_fact(row: Any) -> Dict[str, Any]:
    data = dict(row)
    return {
        "id": data["id"],
        "user_id": data.get("user_id"),
        "subject": data["subject"],
        "predicate": data["predicate"],
        "object": data["object"],
        "valid_from": data["valid_from"],
        "valid_to": data["valid_to"],
        "confidence": data["confidence"],
        "last_updated": data["last_updated"],
        "metadata": json.loads(data["metadata"]) if data.get("metadata") else None,
    }


async def query_facts_at_time(
    subject: Optional[str] = None,
    predicate: Optional[str] = None,
    object_value: Optional[str] = None,
    at: Optional[int] = None,
    min_confidence: float = 0.1,
    user_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    ts = at if at is not None else int(time.time() * 1000)
    clauses = ["(valid_from <= ? AND (valid_to IS NULL OR valid_to >= ?))"]
    params: List[Any] = [ts, ts]
    if user_id:
        clauses.append("user_id=?")
        params.append(user_id)
    if subject:
        clauses.append("subject=?")
        params.append(subject)
    if predicate:
        clauses.append("predicate=?")
        params.append(predicate)
    if object_value:
        clauses.append("object=?")
        params.append(object_value)
    if min_confidence > 0:
        clauses.append("confidence>=?")
        params.append(min_confidence)
    rows = db.fetchall(
        f"SELECT * FROM temporal_facts WHERE {' AND '.join(clauses)} ORDER BY confidence DESC, valid_from DESC",
        tuple(params),
    )
    return [format_fact(row) for row in rows]


async def get_current_fact(
    subject: str,
    predicate: str,
    user_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    clauses = ["subject=?", "predicate=?", "valid_to IS NULL"]
    params: List[Any] = [subject, predicate]
    if user_id:
        clauses.append("user_id=?")
        params.append(user_id)
    row = db.fetchone(
        f"SELECT * FROM temporal_facts WHERE {' AND '.join(clauses)} ORDER BY valid_from DESC LIMIT 1",
        tuple(params),
    )
    return format_fact(row) if row else None


async def query_facts_in_range(
    subject: Optional[str] = None,
    predicate: Optional[str] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
    min_confidence: float = 0.1,
    user_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    clauses: List[str] = []
    params: List[Any] = []
    if start is not None and end is not None:
        clauses.append("((valid_from <= ? AND (valid_to IS NULL OR valid_to >= ?)) OR (valid_from >= ? AND valid_from <= ?))")
        params.extend([end, start, start, end])
    elif start is not None:
        clauses.append("valid_from >= ?")
        params.append(start)
    elif end is not None:
        clauses.append("valid_from <= ?")
        params.append(end)
    if user_id:
        clauses.append("user_id=?")
        params.append(user_id)
    if subject:
        clauses.append("subject=?")
        params.append(subject)
    if predicate:
        clauses.append("predicate=?")
        params.append(predicate)
    if min_confidence > 0:
        clauses.append("confidence>=?")
        params.append(min_confidence)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = db.fetchall(
        f"SELECT * FROM temporal_facts {where} ORDER BY valid_from DESC, confidence DESC",
        tuple(params),
    )
    return [format_fact(row) for row in rows]


async def find_conflicting_facts(
    subject: str,
    predicate: str,
    at: Optional[int] = None,
    user_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    ts = at if at is not None else int(time.time() * 1000)
    clauses = [
        "subject=?",
        "predicate=?",
        "(valid_from <= ? AND (valid_to IS NULL OR valid_to >= ?))",
    ]
    params: List[Any] = [subject, predicate, ts, ts]
    if user_id:
        clauses.append("user_id=?")
        params.append(user_id)
    rows = db.fetchall(
        f"SELECT * FROM temporal_facts WHERE {' AND '.join(clauses)} ORDER BY confidence DESC, valid_from DESC",
        tuple(params),
    )
    return [format_fact(row) for row in rows]


async def get_facts_by_subject(
    subject: str,
    at: Optional[int] = None,
    include_historical: bool = False,
    user_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if include_historical:
        clauses = ["subject=?"]
        params: List[Any] = [subject]
        if user_id:
            clauses.append("user_id=?")
            params.append(user_id)
        rows = db.fetchall(
            f"SELECT * FROM temporal_facts WHERE {' AND '.join(clauses)} ORDER BY predicate ASC, valid_from DESC",
            tuple(params),
        )
    else:
        ts = at if at is not None else int(time.time() * 1000)
        clauses = ["subject=?", "(valid_from <= ? AND (valid_to IS NULL OR valid_to >= ?))"]
        params = [subject, ts, ts]
        if user_id:
            clauses.append("user_id=?")
            params.append(user_id)
        rows = db.fetchall(
            f"SELECT * FROM temporal_facts WHERE {' AND '.join(clauses)} ORDER BY predicate ASC, confidence DESC",
            tuple(params),
        )
    return [format_fact(row) for row in rows]
