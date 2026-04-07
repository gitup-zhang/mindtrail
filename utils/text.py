import re
import hashlib
from typing import List, Set, Dict

SYN_GRPS = [
    ["prefer", "like", "love", "enjoy", "favor", "喜欢", "偏好", "热爱", "喜爱"],
    ["theme", "mode", "style", "layout", "主题", "模式", "风格", "布局"],
    ["meeting", "meet", "session", "call", "sync", "会议", "开会", "沟通"],
    ["dark", "night", "black", "深色", "暗色", "黑色"],
    ["light", "bright", "day", "浅色", "亮色", "明亮"],
    ["user", "person", "people", "customer", "用户", "客户"],
    ["task", "todo", "job", "任务", "待办", "事项"],
    ["note", "memo", "reminder", "笔记", "备忘", "提醒"],
    ["time", "schedule", "when", "date", "时间", "日程", "日期"],
    ["project", "initiative", "plan", "项目", "工程", "计划"],
    ["issue", "problem", "bug", "故障", "报错", "异常"],
    ["document", "doc", "file", "文档", "文件", "资料"],
    ["question", "query", "ask", "提问", "疑问", "问题"],
    ["memory", "remember", "recall", "记忆", "记住", "回忆"],
    ["search", "retrieve", "query", "搜索", "检索", "查询"],
    ["summary", "insight", "reflect", "总结", "洞察", "反思"],
    ["build", "create", "make", "构建", "创建", "搭建"],
]

CMAP: Dict[str, str] = {}
SLOOK: Dict[str, Set[str]] = {}

for grp in SYN_GRPS:
    can = grp[0]
    sset = set(grp)
    for w in grp:
        CMAP[w] = can
        SLOOK[can] = sset

STEM_RULES = [
    (r"ies$", "y"),
    (r"ing$", ""),
    (r"ers?$", "er"),
    (r"ed$", ""),
    (r"s$", ""),
]

CJK_PAT = r"\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af"
TOK_PAT = re.compile(rf"[a-z0-9]+|[{CJK_PAT}]+", re.I)
CJK_RE = re.compile(rf"[{CJK_PAT}]")
CJK_RUN_RE = re.compile(rf"[{CJK_PAT}]+")


def _expand_cjk_token(tok: str) -> List[str]:
    if len(tok) <= 1:
        return [tok]

    expanded: List[str] = []
    max_n = min(4, len(tok))
    for size in range(2, max_n + 1):
        for index in range(len(tok) - size + 1):
            expanded.append(tok[index : index + size])

    if len(tok) <= 6:
        expanded.append(tok)

    seen: Set[str] = set()
    ordered: List[str] = []
    for item in expanded:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered

def tokenize(text: str) -> List[str]:
    res: List[str] = []
    for tok in TOK_PAT.findall(text):
        low = tok.lower()
        if re.fullmatch(rf"[{CJK_PAT}]+", tok):
            res.extend(_expand_cjk_token(tok))
        else:
            res.append(low)
    return res

def stem(tok: str) -> str:
    if len(tok) <= 3: return tok
    for pat, rep in STEM_RULES:
        if re.search(pat, tok):
            st = re.sub(pat, rep, tok)
            if len(st) >= 3: return st
    return tok

def canonicalize_token(tok: str) -> str:
    if not tok: return ""
    low = tok.lower()
    if low in CMAP: return CMAP[low]
    st = stem(low)
    return CMAP.get(st, st)

def canonical_tokens_from_text(text: str) -> List[str]:
    res = []
    for tok in tokenize(text):
        can = canonicalize_token(tok)
        if can and len(can) > 1:
            res.append(can)
    return res

def synonyms_for(tok: str) -> Set[str]:
    can = canonicalize_token(tok)
    return SLOOK.get(can, {can})

def build_search_doc(text: str) -> str:
    can = canonical_tokens_from_text(text)
    exp = set()
    for tok in can:
        exp.add(tok)
        syns = SLOOK.get(tok)
        if syns:
            exp.update(syns)
    return " ".join(exp)

def build_fts_query(text: str) -> str:
    can = canonical_tokens_from_text(text)
    if not can: return ""
    uniq = sorted(list(set(t for t in can if len(t) > 1)))
    return " OR ".join(f'"{t}"' for t in uniq)

def canonical_token_set(text: str) -> Set[str]:
    return set(canonical_tokens_from_text(text))


def contains_cjk(text: str) -> bool:
    return bool(CJK_RE.search(text))


def extract_cjk_ngrams(text: str, min_n: int = 2, max_n: int = 4) -> List[str]:
    grams: List[str] = []
    for run in CJK_RUN_RE.findall(text):
        upper = min(max_n, len(run))
        for size in range(min_n, upper + 1):
            for index in range(len(run) - size + 1):
                grams.append(run[index : index + size])
    return grams


def stable_text_fallback_hash(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()
