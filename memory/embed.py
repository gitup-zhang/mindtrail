import math
import time
from typing import Any, Dict, List, Optional

import numpy as np

from ..core.config import env
from ..core.constants import SEC_WTS, SECTOR_CONFIGS
from ..core.db import q
from ..utils.text import canonical_tokens_from_text, canonicalize_token, contains_cjk, extract_cjk_ngrams, synonyms_for


def _fnv1a(value: str) -> int:
    hash_value = 0x811C9DC5
    for char in value:
        hash_value = (hash_value ^ ord(char)) * 16777619
        hash_value &= 0xFFFFFFFF
    return hash_value


def _murmurish(value: str, seed: int) -> int:
    hash_value = seed
    for char in value:
        hash_value = (hash_value ^ ord(char)) * 0x5BD1E995
        hash_value &= 0xFFFFFFFF
        hash_value = (hash_value >> 13) ^ hash_value
        hash_value &= 0xFFFFFFFF
    return hash_value


def _add_feature(vector: np.ndarray, key: str, weight: float) -> None:
    first = _fnv1a(key)
    second = _murmurish(key, 0xDEADBEEF)
    signed_weight = weight * (1.0 - float((first & 1) << 1))
    dim = len(vector)

    if (dim & (dim - 1)) == 0:
        vector[first & (dim - 1)] += signed_weight
        vector[second & (dim - 1)] += signed_weight * 0.5
    else:
        vector[first % dim] += signed_weight
        vector[second % dim] += signed_weight * 0.5


def _add_positional_feature(vector: np.ndarray, position: int, weight: float) -> None:
    dim = len(vector)
    index = position % dim
    angle = position / pow(10000, (2 * index) / dim)
    vector[index] += weight * math.sin(angle)
    vector[(index + 1) % dim] += weight * math.cos(angle)


def _generate_synthetic_embedding(text: str, sector: str) -> List[float]:
    vector = np.zeros(env.vec_dim, dtype=np.float32)
    tokens = canonical_tokens_from_text(text)
    if not tokens:
        return (np.ones(env.vec_dim, dtype=np.float32) / math.sqrt(env.vec_dim)).tolist()

    expanded: List[str] = []
    for token in tokens:
        expanded.append(token)
        for synonym in synonyms_for(token):
            expanded.append(canonicalize_token(synonym))

    weighted_terms: Dict[str, int] = {}
    for token in expanded:
        weighted_terms[token] = weighted_terms.get(token, 0) + 1

    sector_weight = SEC_WTS.get(sector, 1.0)
    total_terms = max(1, len(expanded))

    for token, count in weighted_terms.items():
        tf = count / total_terms
        idf = math.log(1 + total_terms / count)
        token_weight = (tf * idf + 1) * sector_weight
        _add_feature(vector, f"{sector}|tok|{token}", token_weight)

        if len(token) >= 3:
            for index in range(len(token) - 2):
                trigram = token[index : index + 3]
                _add_feature(vector, f"{sector}|tri|{trigram}", token_weight * 0.4)

    for index in range(len(tokens) - 1):
        pair = f"{tokens[index]}_{tokens[index + 1]}"
        _add_feature(vector, f"{sector}|bi|{pair}", 1.2 * sector_weight / (1 + index * 0.1))

    if contains_cjk(text):
        _add_feature(vector, f"{sector}|lang|cjk", 0.3 * sector_weight)
        for gram in extract_cjk_ngrams(text, 2, 4):
            _add_feature(vector, f"{sector}|cjk|{gram}", 0.7 * sector_weight / math.sqrt(len(gram)))

    for index in range(min(len(tokens), 50)):
        _add_positional_feature(vector, index, 0.5 * sector_weight / math.log1p(total_terms))

    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm
    return vector.tolist()


async def embed_for_sector(text: str, sector: str) -> List[float]:
    if sector not in SECTOR_CONFIGS:
        raise ValueError(f"Unknown sector: {sector}")
    return _generate_synthetic_embedding(text, sector)


async def embed_multi_sector(
    memory_id: str,
    text: str,
    sectors: List[str],
    chunks: Optional[List[dict]] = None,
) -> List[Dict[str, Any]]:
    del chunks
    q.ins_log(entry_id=memory_id, model="synthetic-multi-sector", status="pending", ts=int(time.time() * 1000))
    results: List[Dict[str, Any]] = []

    try:
        for sector in sectors:
            vector = await embed_for_sector(text, sector)
            results.append({"sector": sector, "vector": vector, "dim": len(vector)})
        q.upd_log(entry_id=memory_id, status="completed")
        return results
    except Exception as exc:
        q.upd_log(entry_id=memory_id, status="failed", err=str(exc))
        raise


def calc_mean_vec(embedding_results: List[Dict[str, Any]], all_sectors: List[str]) -> List[float]:
    del all_sectors
    if not embedding_results:
        return []

    dimension = embedding_results[0]["dim"]
    mean = np.zeros(dimension, dtype=np.float32)
    for item in embedding_results:
        mean += np.array(item["vector"], dtype=np.float32)
    mean /= len(embedding_results)
    return mean.tolist()
