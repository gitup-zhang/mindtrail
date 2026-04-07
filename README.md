# MindTrail

## What It Does

MindTrail is a long-term memory engine for LLM agents.

It stores input as retrievable long-term memory and provides:

- `root-child` ingestion for long documents
- multi-sector memory classification
- hybrid retrieval with waypoint expansion
- salience reinforcement, decay, and reflection
- temporal facts and timeline queries
- Chinese and English retrieval support

## Project Structure

```text
README.md
requirements.txt
demo.py
.gitignore
.gitattributes
__init__.py
main.py
core/
memory/
ops/
temporal_graph/
utils/
migrations/
```

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the demo:

```bash
python demo.py
```

Notes:

- SQLite is used by default
- `demo.py` shows ingestion, retrieval, reflection, decay, and timeline queries
