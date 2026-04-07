import struct
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .db import db


@dataclass
class VectorRow:
    id: str
    sector: str
    vector: List[float]
    dim: int


class SQLiteVectorStore:
    async def storeVector(
        self,
        memory_id: str,
        sector: str,
        vector: List[float],
        dim: int,
        user_id: Optional[str] = None,
    ) -> None:
        blob = struct.pack(f"{len(vector)}f", *vector)
        db.execute(
            "INSERT OR REPLACE INTO vectors (id, sector, user_id, v, dim) VALUES (?, ?, ?, ?, ?)",
            (memory_id, sector, user_id, blob, dim),
        )
        db.commit()

    async def getVectorsById(self, memory_id: str) -> List[VectorRow]:
        rows = db.fetchall("SELECT * FROM vectors WHERE id=?", (memory_id,))
        return [self._row_to_vector(row) for row in rows]

    async def getVector(self, memory_id: str, sector: str) -> Optional[VectorRow]:
        row = db.fetchone("SELECT * FROM vectors WHERE id=? AND sector=?", (memory_id, sector))
        return self._row_to_vector(row) if row else None

    async def deleteVectors(self, memory_id: str) -> None:
        db.execute("DELETE FROM vectors WHERE id=?", (memory_id,))
        db.commit()

    async def search(
        self,
        vector: List[float],
        sector: str,
        k: int,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        params: list[Any] = [sector]
        sql = "SELECT id, v FROM vectors WHERE sector=?"
        if filter and filter.get("user_id"):
            sql += " AND user_id=?"
            params.append(filter["user_id"])

        rows = db.fetchall(sql, tuple(params))
        query_vec = np.array(vector, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        results: List[Dict[str, Any]] = []

        for row in rows:
            dim = len(row["v"]) // 4
            candidate = np.array(struct.unpack(f"{dim}f", row["v"]), dtype=np.float32)
            denom = query_norm * np.linalg.norm(candidate)
            similarity = float(np.dot(query_vec, candidate) / denom) if denom > 0 else 0.0
            results.append({"id": row["id"], "similarity": similarity})

        results.sort(key=lambda item: item["similarity"], reverse=True)
        return results[:k]

    def _row_to_vector(self, row: Any) -> VectorRow:
        dim = len(row["v"]) // 4
        vector = list(struct.unpack(f"{dim}f", row["v"]))
        return VectorRow(id=row["id"], sector=row["sector"], vector=vector, dim=row["dim"])


vector_store = SQLiteVectorStore()
