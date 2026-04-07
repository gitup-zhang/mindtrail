import math
import time
from typing import Callable, Dict, Optional

from ..core.db import db, q
from ..utils.vectors import buf_to_vec, vec_to_buf

_active_queries = 0


def inc_q() -> None:
    global _active_queries
    _active_queries += 1


def dec_q() -> None:
    global _active_queries
    _active_queries = max(0, _active_queries - 1)


def calc_recency_score(last_seen: int) -> float:
    age_days = max(0.0, (time.time() * 1000 - last_seen) / 86_400_000.0)
    return math.exp(-age_days / 7.0) * (1.0 - min(age_days / 60.0, 1.0))


def pick_tier(memory: Dict[str, object], now_ts: int) -> str:
    salience = float(memory.get("salience") or 0.0)
    updated_at = int(memory.get("last_seen_at") or memory.get("updated_at") or now_ts)
    age_days = max(0.0, (now_ts - updated_at) / 86_400_000.0)
    if salience >= 0.7 and age_days <= 7:
        return "hot"
    if salience >= 0.4 or age_days <= 21:
        return "warm"
    return "cold"


def _compress_vector(vector: list[float], factor: float, min_dim: int = 64) -> list[float]:
    if len(vector) <= min_dim:
        return vector

    target_dim = max(min_dim, min(len(vector), int(len(vector) * factor)))
    if target_dim >= len(vector):
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


async def apply_decay(user_id: Optional[str] = None, limit: int = 100) -> Dict[str, object]:
    now = int(time.time() * 1000)
    rows = q.all_mem_by_user(user_id, limit, 0) if user_id else q.all_mem(limit, 0)

    processed = 0
    updated = 0
    compressed = 0
    tiers = {"hot": 0, "warm": 0, "cold": 0}

    for row in rows:
        memory = dict(row)
        processed += 1
        tier = pick_tier(memory, now)
        tiers[tier] += 1

        age_days = max(0.0, (now - int(memory["last_seen_at"] or memory["updated_at"] or now)) / 86_400_000.0)
        decay_lambda = {"hot": 0.005, "warm": 0.02, "cold": 0.05}[tier]
        current_salience = float(memory["salience"] or 0.5)
        new_salience = max(0.0, min(1.0, current_salience * math.exp(-decay_lambda * age_days)))

        memory_updated = False
        if abs(new_salience - current_salience) > 0.001:
            db.execute(
                "UPDATE memories SET salience=?, updated_at=? WHERE id=?",
                (new_salience, now, memory["id"]),
            )
            memory_updated = True

        if tier == "cold" and memory.get("mean_vec"):
            original = buf_to_vec(memory["mean_vec"])
            compact = _compress_vector(original, 0.25)
            if len(compact) < len(original):
                db.execute(
                    "UPDATE memories SET compressed_vec=?, updated_at=? WHERE id=?",
                    (vec_to_buf(compact), now, memory["id"]),
                )
                compressed += 1
                memory_updated = True

        if memory_updated:
            updated += 1

    db.commit()
    return {
        "processed": processed,
        "updated": updated,
        "compressed": compressed,
        "tiers": tiers,
        "user_id": user_id,
    }


async def on_query_hit(
    memory_id: str,
    sector: str,
    reembed_fn: Optional[Callable[[str], object]] = None,
) -> None:
    del sector
    del reembed_fn
    row = q.get_mem(memory_id)
    if not row:
        return

    current_salience = float(row["salience"] or 0.5)
    boosted_salience = min(1.0, current_salience + 0.08)
    now = int(time.time() * 1000)
    db.execute(
        "UPDATE memories SET salience=?, last_seen_at=?, updated_at=? WHERE id=?",
        (boosted_salience, now, now, memory_id),
    )
    db.commit()
