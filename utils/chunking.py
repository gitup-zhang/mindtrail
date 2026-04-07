import math
import re
from typing import List, Dict, TypedDict

class Chunk(TypedDict):
    text: str
    start: int
    end: int
    tokens: int

CPT = 4
CJK_PAT = r"\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af"
CJK_RE = re.compile(rf"[{CJK_PAT}]")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?。！？；;])\s*")


def est(t: str) -> int:
    cjk_chars = len(CJK_RE.findall(t))
    other_chars = max(0, len(t) - cjk_chars)
    return max(1, math.ceil(cjk_chars * 0.7 + other_chars / CPT))


def _join_text(left: str, right: str) -> str:
    if not left:
        return right
    if not right:
        return left
    if CJK_RE.search(left[-1]) and CJK_RE.match(right[0]):
        return left + right
    return left + " " + right

def chunk_text(txt: str, tgt: int = 768, ovr: float = 0.1) -> List[Chunk]:
    tot = est(txt)
    if tot <= tgt:
        return [{"text": txt, "start": 0, "end": len(txt), "tokens": tot}]

    tch = tgt * CPT
    och = math.floor(tch * ovr)
    paras = re.split(r"\n\n+", txt)

    chks: List[Chunk] = []
    cur = ""
    cs = 0

    for p in paras:
        sents = [segment for segment in SENT_SPLIT_RE.split(p) if segment]
        for s in sents:
            pot = _join_text(cur, s)
            if len(pot) > tch and len(cur) > 0:
                current_len = len(cur)
                chks.append({
                    "text": cur,
                    "start": cs,
                    "end": cs + len(cur),
                    "tokens": est(cur)
                })
                ovt = cur[-och:] if och < len(cur) else cur
                cs = cs + current_len - len(ovt)
                cur = _join_text(ovt, s)
            else:
                cur = pot

    if len(cur) > 0:
        chks.append({
            "text": cur,
            "start": cs,
            "end": cs + len(cur),
            "tokens": est(cur)
        })
    return chks

def agg_vec(vecs: List[List[float]]) -> List[float]:
    n = len(vecs)
    if not n: raise ValueError("no vecs")
    if n == 1: return vecs[0].copy()

    d = len(vecs[0])
    r = [0.0] * d
    for v in vecs:
        for i in range(d):
            r[i] += v[i]

    rc = 1.0 / n
    for i in range(d):
        r[i] *= rc
    return r

def join_chunks(cks: List[Chunk]) -> str:
    return " ".join(c["text"] for c in cks) if cks else ""
