CREATE TABLE IF NOT EXISTS query (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT UNIQUE,
    datetime TEXT
);

CREATE TABLE IF NOT EXISTS paper_search (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT,
    pdf_url TEXT,
    filename TEXT,
    pdf_data BLOB
);

CREATE TABLE IF NOT EXISTS document_page (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_search_id INTEGER,
    page_number INTEGER,
    text TEXT,
    FOREIGN KEY (paper_search_id) REFERENCES paper_search(id)
);

CREATE TABLE IF NOT EXISTS document_text (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_search_id INTEGER,
    text TEXT,
    FOREIGN KEY (paper_search_id) REFERENCES paper_search(id)
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
    faiss_index_id INTEGER,
    document_page_split_id INTEGER,
    embedding BLOB,
    FOREIGN KEY (document_page_split_id) REFERENCES document_page_split(id)
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

CREATE TABLE IF NOT EXISTS search_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT,
    distance REAL,
    faiss_index INTEGER,
    document_text TEXT,
    document_index INTEGER,
    document_page_id INTEGER,
    FOREIGN KEY (document_page_id) REFERENCES document_page(id)
);

CREATE TABLE IF NOT EXISTS queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    original_query TEXT,
    prompt TEXT,
    generated_query TEXT
);

