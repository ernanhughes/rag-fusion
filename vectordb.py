import sqlite3

import numpy as np
import sqlite_vec

from config import appConfig
from markdowncleaner import MarkdownCleaner
from ollama_utils import generate_embeddings, to_embedding
from sqlite_vec import serialize_float32


class VectorDB:
    def __init__(self, config=appConfig):
        self.config = config
        self.conn = sqlite3.connect(config["VECTOR_DB_NAME"])
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)
        self.cursor = self.conn.cursor()
        self.create_tables()

    def create_tables(self):
        self.cursor.execute("DROP TABLE IF EXISTS pdf_vec;")
        sql = f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS pdf_vec USING vec0(
                    embedding float[{self.config["EMBEDDING_DIMS"]}], 
                    id TEXT);
             """
        print(sql)
        self.cursor.execute(sql)
        self.cursor.execute(
            'CREATE TABLE IF NOT EXISTS pdf_lookup (id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT);')
        self.conn.commit()

    def load_documents_from_db(self):
        """Load documents from the downloads database table."""
        conn = sqlite3.connect(self.config["DB_NAME"])
        cursor = conn.cursor()
        cursor.execute("SELECT id, text from document_page_split")
        documents = cursor.fetchall()
        conn.close()
        return [(doc[0], MarkdownCleaner.clean_markdown(doc[1])) for doc in documents]

    def search(self, fts_search_query: str, top_k: int = 2):
        fts_results = self.cursor.execute("SELECT id FROM pdf_fts WHERE pdf_fts MATCH ? ORDER BY rank LIMIT 5",
                                          (fts_search_query,)).fetchall()
        query_embedding = generate_embeddings(fts_search_query)
        print(f"Query embedding: {query_embedding}")
        vec_results = self.cursor.execute(
            "SELECT rowid, distance FROM pdf_vec WHERE embedding MATCH ? AND K = ? ORDER BY distance",
            [np.array(query_embedding, dtype=np.float32).tobytes(), top_k]).fetchall()

        combined_results = VectorDB.reciprocal_rank_fusion(fts_results, vec_results)
        for id, score in combined_results:
            print(f'ID: {id}, Content: {self.lookup_row(id)}, RRF Score: {score}')

    @staticmethod
    def serialize_f32(vec):
        return np.array(vec, dtype=np.float32).tobytes()

    @staticmethod
    def reciprocal_rank_fusion(fts_results, vec_results, k=60):
        rank_dict = {}

        # Process FTS results
        for rank, (id,) in enumerate(fts_results):
            if id not in rank_dict:
                rank_dict[id] = 0
            rank_dict[id] += 1 / (k + rank + 1)

        # Process vector results
        for rank, (rowid, distance) in enumerate(vec_results):
            if rowid not in rank_dict:
                rank_dict[rowid] = 0
            rank_dict[rowid] += 1 / (k + rank + 1)

        # Sort by RRF score
        sorted_results = sorted(rank_dict.items(), key=lambda x: x[1], reverse=True)
        return sorted_results

    @staticmethod
    def or_words(input_string):
        # Split the input string into words
        words = input_string.split()

        # Join the words with ' OR ' in between
        result = ' OR '.join(words)

        return result

    def lookup_row(self, id):
        row_lookup = self.cursor.execute('''
        SELECT content FROM pdf_lookup WHERE id = ?
        ''', (id,)).fetchall()
        content = ''
        for row in row_lookup:
            content = row[0]
            break
        return content

    def close(self):
        self.conn.close()


vec = VectorDB()
docs = vec.load_documents_from_db()
for doc in docs:
    embedded_text = to_embedding(doc[1])
    print(f"Embedding text: {embedded_text}")
    embed_float32 = serialize_float32(embedded_text)
    print(f"Embedding text: {embed_float32}")
    id = str(doc[1])
    vec.cursor.execute('''
    INSERT INTO pdf_lookup (content, id) VALUES (?, ?)
    ''', (embed_float32, id))
    print(f"Inserted  Embedding: {doc[0]}")
vec.conn.commit()
