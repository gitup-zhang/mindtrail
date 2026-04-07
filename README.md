# MindTrail
注：该记忆引擎是临时的简单开源版本，主要用来展示系统相关功能，详细版本因老师要求并未开源，但也实现了绝大部分功能

面向 LLM Agent 的长期记忆引擎。集成多扇区向量检索 (HSG)、关联图扩展 (Waypoint)、时态事实知识库 (Temporal Fact Graph)，以及自动衰减与反思机制——全部基于 SQLite，零外部 API 依赖。


## 特性

- **多扇区记忆分类** — 将每条记忆自动归入 5 个认知扇区（语义 / 程序性 / 情景 / 反思 / 情感），独立生成向量嵌入
- **混合检索评分** — 融合向量相似度、词元重叠、关键词匹配、标签匹配、Waypoint 权重与时间衰减
- **Waypoint 关联图** — 基于向量相似度自动构建记忆间链路，低匹配时通过 BFS 扩展召回
- **时态事实图** — SPO 三元组 + 有效区间 `[valid_from, valid_to)`，支持时间点快照、范围查询、冲突检测和时间线 diff
- **三级衰减** — Hot / Warm / Cold 分级指数衰减，冷记忆自动压缩向量维度
- **自动反思** — 混合相似度（55% 向量 + 45% 词汇）聚类，生成反思摘要并标记源记忆为已合并
- **文档摄入** — 支持纯文本 / PDF / DOCX / HTML / URL，大文档自动拆分为 root-child 结构
- **合成嵌入** — 基于 FNV1a + TF-IDF + trigram + 位置编码的自研嵌入，无需调用外部模型

## 快速开始

```bash
pip install -r requirements.txt
python demo.py
```

### 代码示例

```python
import asyncio
from mindtrail import MindTrail

async def main():
    engine = MindTrail(user="alice")

    # 存储记忆
    await engine.add("FastAPI 底层使用 Starlette 处理路由", tags=["python"])

    # 语义检索
    results = await engine.search("FastAPI 如何处理路由？", limit=5)
    for r in results:
        print(f"{r['score']:.3f} | {r['content']}")

    # 存储 & 查询时态事实
    await engine.remember_fact("Alice", "就职于", "Acme 公司")
    facts = await engine.facts_at(subject="Alice")

    # 对比两个时间点的事实变化
    diff = await engine.compare_time_points("Alice", t1, t2)

    # 维护
    await engine.decay()    # 三级衰减
    await engine.reflect()  # 自动反思

asyncio.run(main())
```

## 架构

```
mindtrail/
├── core/
│   ├── config.py            # 配置（环境变量）
│   ├── constants.py         # 扇区定义 & 模式匹配
│   ├── db.py                # SQLite 封装 + 自动迁移
│   └── vector_store.py      # 向量存储 & 余弦相似度检索
├── memory/
│   ├── engine.py            # HSG 核心（分类 → 嵌入 → 评分 → Waypoint）
│   ├── embed.py             # 合成嵌入生成
│   ├── decay.py             # 三级衰减 & 向量压缩
│   └── reflection_core.py   # 聚类 & 反思记忆生成
├── ops/
│   ├── ingestion.py         # 文档摄入（root-child 拆分）
│   └── extractor.py         # 多格式文本提取
├── temporal_graph/
│   ├── store_core.py        # 事实 CRUD & 版本管理
│   ├── query_core.py        # 点查询 / 范围 / 冲突 / 主体
│   └── timeline_core.py     # 时间线 & 快照对比
├── utils/                   # 文本、分块、关键词、向量工具
├── migrations/              # SQL 迁移脚本
├── main.py                  # MindTrail 统一入口
└── demo.py                  # 演示脚本
```

## 技术栈

| 组件 | 选型 |
| ---- | ---- |
| 语言 | Python 3.11+，全链路 async/await |
| 存储 | SQLite（WAL 模式） |
| 向量 | NumPy 余弦相似度 |
| 嵌入 | 自研合成方案，零外部依赖 |
| 文档解析 | pypdf · mammoth · markdownify · httpx |

## License

MIT
