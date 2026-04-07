from typing import Any, Dict, List, Optional

from .core.db import db, q
from .memory.decay import apply_decay
from .memory.engine import hsg_query
from .memory.reflection_core import run_reflection
from .ops.ingestion import ingest_document
from .temporal_graph.query_core import (
    find_conflicting_facts,
    get_current_fact,
    get_facts_by_subject,
    query_facts_at_time,
    query_facts_in_range,
)
from .temporal_graph.store_core import insert_fact
from .temporal_graph.timeline_core import compare_time_points as compare_temporal_points
from .temporal_graph.timeline_core import get_subject_timeline


class MindTrail:
    def __init__(self, user: Optional[str] = None):
        self.default_user = user
        db.connect()

    async def add(
        self,
        content: str,
        user_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        uid = user_id or self.default_user
        result = await ingest_document("text", content, meta=meta, user_id=uid, tags=tags)
        if "root_memory_id" in result:
            result["id"] = result["root_memory_id"]
        return result

    async def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        limit: int = 10,
        **filters: Any,
    ) -> List[Dict[str, Any]]:
        scoped_filters = dict(filters)
        scoped_filters["user_id"] = user_id or self.default_user
        return await hsg_query(query, limit, scoped_filters)

    async def get(self, memory_id: str) -> Optional[Dict[str, Any]]:
        row = q.get_mem(memory_id)
        return dict(row) if row else None

    async def delete(self, memory_id: str) -> None:
        q.del_mem(memory_id)

    async def delete_all(self, user_id: Optional[str] = None) -> None:
        uid = user_id or self.default_user
        if uid:
            db.execute(
                """
                DELETE FROM temporal_edges
                WHERE source_id IN (SELECT id FROM temporal_facts WHERE user_id=?)
                   OR target_id IN (SELECT id FROM temporal_facts WHERE user_id=?)
                """,
                (uid, uid),
            )
            db.execute("DELETE FROM temporal_facts WHERE user_id=?", (uid,))
            q.del_mem_by_user(uid)

    def history(
        self,
        user_id: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        uid = user_id or self.default_user
        rows = q.all_mem_by_user(uid, limit, offset) if uid else q.all_mem(limit, offset)
        return [dict(row) for row in rows]

    async def remember_fact(
        self,
        subject: str,
        predicate: str,
        object_value: str,
        valid_from: Optional[int] = None,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> str:
        uid = user_id or self.default_user
        return await insert_fact(
            subject=subject,
            predicate=predicate,
            object_value=object_value,
            valid_from=valid_from,
            confidence=confidence,
            metadata=metadata,
            user_id=uid,
        )

    async def facts_at(
        self,
        at: Optional[int] = None,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object_value: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return await query_facts_at_time(
            subject=subject,
            predicate=predicate,
            object_value=object_value,
            at=at,
            user_id=user_id or self.default_user,
        )

    async def subject_facts(
        self,
        subject: str,
        include_historical: bool = False,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return await get_facts_by_subject(
            subject,
            include_historical=include_historical,
            user_id=user_id or self.default_user,
        )

    async def subject_timeline(
        self,
        subject: str,
        predicate: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return await get_subject_timeline(
            subject,
            predicate=predicate,
            user_id=user_id or self.default_user,
        )

    async def current_fact(
        self,
        subject: str,
        predicate: str,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        return await get_current_fact(subject, predicate, user_id=user_id or self.default_user)

    async def facts_in_range(
        self,
        start: Optional[int] = None,
        end: Optional[int] = None,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return await query_facts_in_range(
            subject=subject,
            predicate=predicate,
            start=start,
            end=end,
            user_id=user_id or self.default_user,
        )

    async def conflicting_facts(
        self,
        subject: str,
        predicate: str,
        at: Optional[int] = None,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return await find_conflicting_facts(
            subject=subject,
            predicate=predicate,
            at=at,
            user_id=user_id or self.default_user,
        )

    async def compare_time_points(
        self,
        subject: str,
        first_at: int,
        second_at: int,
        user_id: Optional[str] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        return await compare_temporal_points(
            subject=subject,
            first_at=first_at,
            second_at=second_at,
            user_id=user_id or self.default_user,
        )

    async def decay(
        self,
        user_id: Optional[str] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        return await apply_decay(user_id=user_id or self.default_user, limit=limit)

    async def reflect(
        self,
        user_id: Optional[str] = None,
        limit: int = 100,
        min_cluster_size: int = 2,
        similarity_threshold: float = 0.28,
    ) -> Dict[str, Any]:
        return await run_reflection(
            user_id=user_id or self.default_user,
            limit=limit,
            min_cluster_size=min_cluster_size,
            similarity_threshold=similarity_threshold,
        )


MemoryEngine = MindTrail
