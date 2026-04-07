import json
from typing import Any, Dict, List, Optional

from ..core.db import db


def _row_to_fact(row: Any) -> Dict[str, Any]:
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
        "metadata": json.loads(data["metadata"]) if data["metadata"] else None,
    }


async def get_subject_timeline(
    subject: str,
    predicate: Optional[str] = None,
    user_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    clauses = ["subject=?"]
    params: List[Any] = [subject]
    if predicate:
        clauses.append("predicate=?")
        params.append(predicate)
    if user_id:
        clauses.append("user_id=?")
        params.append(user_id)
    rows = db.fetchall(
        f"SELECT * FROM temporal_facts WHERE {' AND '.join(clauses)} ORDER BY valid_from ASC",
        tuple(params),
    )

    timeline: List[Dict[str, Any]] = []
    for row in rows:
        timeline.append(
            {
                "timestamp": row["valid_from"],
                "subject": row["subject"],
                "predicate": row["predicate"],
                "object": row["object"],
                "confidence": row["confidence"],
                "change_type": "created",
            }
        )
        if row["valid_to"]:
            timeline.append(
                {
                    "timestamp": row["valid_to"],
                    "subject": row["subject"],
                    "predicate": row["predicate"],
                    "object": row["object"],
                    "confidence": row["confidence"],
                    "change_type": "invalidated",
                }
            )

    timeline.sort(key=lambda item: item["timestamp"])
    return timeline


async def compare_time_points(
    subject: str,
    first_at: int,
    second_at: int,
    user_id: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    clauses = ["subject=?", "(valid_from <= ? AND (valid_to IS NULL OR valid_to >= ?))"]
    if user_id:
        clauses.append("user_id=?")

    params_first: List[Any] = [subject, first_at, first_at]
    params_second: List[Any] = [subject, second_at, second_at]
    if user_id:
        params_first.append(user_id)
        params_second.append(user_id)

    sql = f"SELECT * FROM temporal_facts WHERE {' AND '.join(clauses)}"
    first_rows = db.fetchall(sql, tuple(params_first))
    second_rows = db.fetchall(sql, tuple(params_second))

    first_map = {row["predicate"]: row for row in first_rows}
    second_map = {row["predicate"]: row for row in second_rows}

    added: List[Dict[str, Any]] = []
    removed: List[Dict[str, Any]] = []
    changed: List[Dict[str, Any]] = []
    unchanged: List[Dict[str, Any]] = []

    for predicate, current in second_map.items():
        previous = first_map.get(predicate)
        if not previous:
            added.append(_row_to_fact(current))
        elif previous["object"] != current["object"] or previous["id"] != current["id"]:
            changed.append({"before": _row_to_fact(previous), "after": _row_to_fact(current)})
        else:
            unchanged.append(_row_to_fact(current))

    for predicate, previous in first_map.items():
        if predicate not in second_map:
            removed.append(_row_to_fact(previous))

    return {
        "added": added,
        "removed": removed,
        "changed": changed,
        "unchanged": unchanged,
    }
