from typing import Dict, List, TypedDict, Pattern
import re

class SectorCfg(TypedDict):
    model: str
    decay_lambda: float
    weight: float
    patterns: List[Pattern]

SECTOR_CONFIGS: Dict[str, SectorCfg] = {
    "episodic": {
        "model": "episodic-optimized",
        "decay_lambda": 0.015,
        "weight": 1.2,
        "patterns": [
            re.compile(r"\b(today|yesterday|tomorrow|last\s+(week|month|year)|next\s+(week|month|year))\b", re.I),
            re.compile(r"\b(remember\s+when|recall|that\s+time|when\s+I|I\s+was|we\s+were)\b", re.I),
            re.compile(r"\b(went|saw|met|felt|heard|visited|attended|participated)\b", re.I),
            re.compile(r"\b(at\s+\d{1,2}:\d{2}|on\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b", re.I),
            re.compile(r"\b(event|moment|experience|incident|occurrence|happened)\b", re.I),
            re.compile(r"\bI\s+'?m\s+going\s+to\b", re.I),
            re.compile(r"(今天|昨天|明天|刚才|刚刚|之前|之后|上周|下周|上个月|去年|明年)"),
            re.compile(r"(记得|回忆|那次|当时|那天|我当时|我们当时|后来)"),
            re.compile(r"(去了|看到|见到|参加了|遇到|发生了|经历了|听到|感受到)"),
            re.compile(r"(\d{1,2}点\d{0,2}分?|周一|周二|周三|周四|周五|周六|周日|星期[一二三四五六日天])"),
            re.compile(r"(事件|瞬间|经历|事故|事情|发生)"),
        ],
    },
    "semantic": {
        "model": "semantic-optimized",
        "decay_lambda": 0.005,
        "weight": 1.0,
        "patterns": [
            re.compile(r"\b(is\s+a|represents|means|stands\s+for|defined\s+as)\b", re.I),
            re.compile(r"\b(concept|theory|principle|law|hypothesis|theorem|axiom)\b", re.I),
            re.compile(r"\b(fact|statistic|data|evidence|proof|research|study|report)\b", re.I),
            re.compile(r"\b(capital|population|distance|weight|height|width|depth)\b", re.I),
            re.compile(r"\b(history|science|geography|math|physics|biology|chemistry)\b", re.I),
            re.compile(r"\b(know|understand|learn|read|write|speak)\b", re.I),
            re.compile(r"(是一种|表示|意味着|定义为|可以理解为)"),
            re.compile(r"(概念|理论|原理|定律|假设|公理|模型|框架)"),
            re.compile(r"(事实|统计|数据|证据|研究|论文|报告|结论)"),
            re.compile(r"(历史|科学|地理|数学|物理|生物|化学|知识)"),
            re.compile(r"(了解|理解|学习|阅读|写作|表达)"),
        ],
    },
    "procedural": {
        "model": "procedural-optimized",
        "decay_lambda": 0.008,
        "weight": 1.1,
        "patterns": [
            re.compile(r"\b(how\s+to|step\s+by\s+step|guide|tutorial|manual|instructions)\b", re.I),
            re.compile(r"\b(first|second|then|next|finally|afterwards|lastly)\b", re.I),
            re.compile(r"\b(install|run|execute|compile|build|deploy|configure|setup)\b", re.I),
            re.compile(r"\b(click|press|type|enter|select|drag|drop|scroll)\b", re.I),
            re.compile(r"\b(method|function|class|algorithm|routine|recipie)\b", re.I),
            re.compile(r"\b(to\s+do|to\s+make|to\s+build|to\s+create)\b", re.I),
            re.compile(r"(如何|怎么|步骤|教程|指南|说明|操作方法|做法)"),
            re.compile(r"(首先|第一步|然后|接着|下一步|最后|完成后)"),
            re.compile(r"(安装|运行|执行|编译|构建|部署|配置|设置)"),
            re.compile(r"(点击|按下|输入|选择|拖动|滚动|复制|粘贴)"),
            re.compile(r"(方法|函数|类|算法|流程|脚本|命令)"),
        ],
    },
    "emotional": {
        "model": "emotional-optimized",
        "decay_lambda": 0.02,
        "weight": 1.3,
        "patterns": [
            re.compile(r"\b(feel|feeling|felt|emotions?|mood|vibe)\b", re.I),
            re.compile(r"\b(happy|sad|angry|mad|excited|scared|anxious|nervous|depressed)\b", re.I),
            re.compile(r"\b(love|hate|like|dislike|adore|detest|enjoy|loathe)\b", re.I),
            re.compile(r"\b(amazing|terrible|awesome|awful|wonderful|horrible|great|bad)\b", re.I),
            re.compile(r"\b(frustrated|confused|overwhelmed|stressed|relaxed|calm)\b", re.I),
            re.compile(r"\b(wow|omg|yay|nooo|ugh|sigh)\b", re.I),
            re.compile(r"[!]{2,}", re.I),
            re.compile(r"(感觉|情绪|心情|状态|氛围)"),
            re.compile(r"(开心|难过|生气|兴奋|害怕|焦虑|紧张|沮丧|崩溃)"),
            re.compile(r"(喜欢|讨厌|热爱|厌烦|享受|受不了)"),
            re.compile(r"(太棒了|太糟了|很好|很差|离谱|烦死了|放松|平静)"),
            re.compile(r"[！]{2,}"),
        ],
    },
    "reflective": {
        "model": "reflective-optimized",
        "decay_lambda": 0.001,
        "weight": 0.8,
        "patterns": [
            re.compile(r"\b(realize|realized|realization|insight|epiphany)\b", re.I),
            re.compile(r"\b(think|thought|thinking|ponder|contemplate|reflect)\b", re.I),
            re.compile(r"\b(understand|understood|understanding|grasp|comprehend)\b", re.I),
            re.compile(r"\b(pattern|trend|connection|link|relationship|correlation)\b", re.I),
            re.compile(r"\b(lesson|moral|takeaway|conclusion|summary|implication)\b", re.I),
            re.compile(r"\b(feedback|review|analysis|evaluation|assessment)\b", re.I),
            re.compile(r"\b(improve|grow|change|adapt|evolve)\b", re.I),
            re.compile(r"(意识到|反思|复盘|洞察|启发|顿悟|想明白了)"),
            re.compile(r"(我觉得|我认为|我发现|我理解到|我总结)"),
            re.compile(r"(规律|趋势|联系|关联|关系|模式)"),
            re.compile(r"(教训|总结|结论|启示|含义)"),
            re.compile(r"(反馈|分析|评估|复审|观察)"),
            re.compile(r"(改进|成长|变化|调整|演化|优化)"),
        ],
    },
}

SEC_WTS = {k: v["weight"] for k, v in SECTOR_CONFIGS.items()}
