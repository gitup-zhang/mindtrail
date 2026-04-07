DROP TABLE IF EXISTS temporal_edges;
DROP TABLE IF EXISTS temporal_facts;

CREATE TABLE IF NOT EXISTS temporal_facts (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    valid_from INTEGER NOT NULL,
    valid_to INTEGER,
    confidence REAL NOT NULL DEFAULT 1.0,
    last_updated INTEGER NOT NULL,
    metadata TEXT
);

CREATE TABLE IF NOT EXISTS temporal_edges (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    valid_from INTEGER NOT NULL,
    valid_to INTEGER,
    weight REAL NOT NULL DEFAULT 1.0,
    metadata TEXT,
    FOREIGN KEY(source_id) REFERENCES temporal_facts(id),
    FOREIGN KEY(target_id) REFERENCES temporal_facts(id)
);

CREATE INDEX IF NOT EXISTS idx_temporal_subject ON temporal_facts(subject);
CREATE INDEX IF NOT EXISTS idx_temporal_predicate ON temporal_facts(predicate);
CREATE INDEX IF NOT EXISTS idx_temporal_valid_from ON temporal_facts(valid_from);
