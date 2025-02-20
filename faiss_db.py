import sqlite3
import re
import numpy as np
import faiss
import ollama

from config import appConfig


class FaissDB:
    def __init__(self, config=appConfig):
        self.config = config
        self.embedding_model = config["EMBEDDING_MODEL"]
        
    def build_index(self, embeddings: np.ndarray) -> faiss.IndexFlatIP:
        dimension = embeddings.shape[1]  # Number of features in each embedding
        print(f"dimension={dimension}")
        # Create a FAISS index
        index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity search
        # Add embeddings to the index
        index.add(embeddings)
        return index
    
    def get_embeddings(self, documents) -> np.ndarray:
        embeddings = []
        for doc in documents:
            print(f"Generating embedding for document {doc[1]}")
            response = ollama.embeddings(model=self.embedding_model, prompt=doc[1])
            embeddings.append(response["embedding"])
        # Convert embeddings to a NumPy array
        embeddings = np.array(embeddings).astype('float32')
        return embeddings


    def get_embedding(self, text) -> np.ndarray:
        response = ollama.embeddings(model=self.embedding_model, prompt=text)
        embeddings = []
        embeddings.append(response["embedding"])
        embeddings = np.array(embeddings).astype('float32')
        return embeddings

    @staticmethod
    def save(index, filename):
        faiss.write_index(index, filename)

    @staticmethod
    def load(filename):
        loaded_index = faiss.read_index(filename)
        return loaded_index
    
    @staticmethod
    def load_documents_from_db(config, max_documents: int = 1000) -> list:
        """Load documents from the downloads database table."""
        conn = sqlite3.connect(config["DB_NAME"])
        cursor = conn.cursor()
        cursor.execute(f"SELECT id, text from document_page_split LIMIT {max_documents}")
        documents = cursor.fetchall()
        conn.close()
        return [(doc[0], FaissDB.clean_text(doc[1])) for doc in documents]

    @staticmethod
    def clean_text(md_text):
        """Remove Markdown syntax and extract clean text."""
        md_text = re.sub(r'\[.*?\]\(.*?\)', '', md_text)  # Remove links
        md_text = re.sub(r'#{1,6}\s*', '', md_text)  # Remove headers
        md_text = re.sub(r'(```.*?```|`.*?`)', '', md_text, flags=re.DOTALL)  # Remove code blocks
        md_text = re.sub(r'\*{1,2}|\_{1,2}', '', md_text)  # Remove bold/italic formatting
        md_text = re.sub(r'>\s*', '', md_text)  # Remove blockquotes
        md_text = re.sub(r'[-+*]\s+', '', md_text)  # Remove bullet points
        md_text = re.sub(r'\d+\.\s+', '', md_text)  # Remove numbered lists
        return md_text.strip()



vec = FaissDB()
docs = vec.load_documents_from_db(30)
print(docs)
embeddings = vec.get_embeddings(docs)
index = vec.build_index(embeddings)


# Query embedding (generate an embedding for the query)
query_text = "RAG Database"
query_embedding = vec.get_embedding(query_text)

# Search for the top-k most similar embeddings
k = 5  # Number of nearest neighbors to retrieve
distances, indices = index.search(query_embedding, k)

# Print results
print("Indices of similar documents:", indices)
print("Distances to similar documents:", distances)
print("Similar documents:")
for i, idx in enumerate(indices[0]):
    print(f"Document {idx}: {docs[idx][1]}")    
