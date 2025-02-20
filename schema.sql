CREATE TABLE IF NOT EXISTS query (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT UNIQUE,
    datetime TEXT
);

CREATE TABLE IF NOT EXISTS arxiv_search (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT,
    pdf_url TEXT,
    filename TEXT,
    pdf_data BLOB
);

CREATE TABLE IF NOT EXISTS document_page (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    arxiv_search_id INTEGER,
    page_number INTEGER,
    text TEXT,
    FOREIGN KEY (arxiv_search_id) REFERENCES arxiv_search(id)
);

CREATE TABLE IF NOT EXISTS document_text (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    arxiv_search_id INTEGER,
    text TEXT,
    FOREIGN KEY (arxiv_search_id) REFERENCES arxiv_search(id)
);

CREATE TABLE IF NOT EXISTS document_page_split (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_page_id INTEGER,
    text TEXT,
    FOREIGN KEY (document_page_id) REFERENCES document_page(id)
);

CREATE TABLE IF NOT EXISTS document_index (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_index INTEGER,
    document_page_id INTEGER,
    text TEXT,
    FOREIGN KEY (document_page_id) REFERENCES document_page(id)
);

CREATE TABLE IF NOT EXISTS document_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_page_id INTEGER,
    text TEXT,
    FOREIGN KEY (document_page_id) REFERENCES document_page(id)
);

CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT,
    generated_queries TEXT,
    standard_response TEXT,
    fusion_response TEXT,
    hierarchical_fusion_response TEXT,
    cross_encoder_reranking_response TEXT,
    multi_stage_fusion_response TEXT,
    evaluation TEXT
);
