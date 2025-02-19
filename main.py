import faiss
import numpy as np
import toml
import PyPDF2
import sqlite3
import nltk
import requests
import os
import xml.etree.ElementTree as ET
from nltk.tokenize import sent_tokenize
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

nltk.download("punkt")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

from config import config
from ollama_utils import generate_embeddings, chat

# Initialize SQLite database
conn = sqlite3.connect("rag_fusion.db")
cursor = conn.cursor()
cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        extracted_text TEXT,
        split_text TEXT
    )
"""
)
cursor.execute(
    """
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
    )
"""
)
cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS arxiv_search (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        query TEXT,
        pdf_url TEXT,
        filename TEXT
    )
"""
)
conn.commit()

# Initialize embeddings model
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


class LLMHandler:
    """Handles all LLM calls in one place."""

    def __init__(self, model):
        self.llm = chat

    def generate_queries(self, original_query):
        prompt = f"Generate {config.config['num_queries']} search queries related to: {original_query}"
        generated_text = self.llm(prompt)
        return generated_text.split("\n")

    def generate_hierarchical_queries(self, original_query):
        prompt = f"Generate broad categories for: {original_query}"
        broad_categories = self.llm(prompt).split("\n")
        sub_queries = []
        for category in broad_categories:
            prompt = f"Generate specific search queries for the category: {category}"
            sub_queries.extend(self.llm(prompt).split("\n"))
        return sub_queries

    def generate_final_response(self, original_query, reranked_docs):
        context = "\n".join(reranked_docs[:5])
        prompt = f"Given the following context:\n{context}\nAnswer the query: {original_query}"
        return self.llm(prompt)

    def evaluate_responses(
        self,
        standard_response,
        fusion_response,
        hierarchical_fusion_response,
        cross_encoder_response,
        multi_stage_response,
    ):
        evaluation_prompt = (
            f"Evaluate the following five responses and assign a grade based on accuracy, relevance, and comprehensiveness.\n\n"
            f"Standard RAG Response:\n{standard_response}\n\n"
            f"RAG-Fusion Response:\n{fusion_response}\n\n"
            f"Hierarchical RAG-Fusion Response:\n{hierarchical_fusion_response}\n\n"
            f"Cross-Encoder Reranking Response:\n{cross_encoder_response}\n\n"
            f"Multi-Stage Fusion Response:\n{multi_stage_response}\n\n"
            f"Provide a rating (1-10) and a brief justification for each response."
        )
        return self.llm(evaluation_prompt)


def multi_stage_fusion(query, vector_store, corpus):
    """Combine BM25 (sparse) and FAISS (dense) retrieval results before fusion."""
    tokenized_corpus = [doc.split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(query.split())

    dense_results = retrieve_documents([query], vector_store)
    combined_results = {
        doc: bm25_scores[i] + dense_results[i][1] for i, doc in enumerate(corpus)
    }

    return sorted(
        combined_results.keys(), key=lambda x: combined_results[x], reverse=True
    )


def evaluate_rag_variants(original_query, vector_store, llm_handler):
    """Evaluate and compare multiple RAG approaches."""

    # Standard RAG
    standard_queries = [original_query]
    standard_docs = [
        doc
        for query in standard_queries
        for doc in retrieve_documents([query], vector_store)
    ]
    standard_response = llm_handler.generate_final_response(
        original_query, standard_docs
    )

    # RAG-Fusion
    fusion_queries = llm_handler.generate_queries(original_query)
    fusion_docs = [
        doc
        for query in fusion_queries
        for doc in retrieve_documents([query], vector_store)
    ]
    fusion_response = llm_handler.generate_final_response(original_query, fusion_docs)

    # Hierarchical RAG-Fusion
    hierarchical_queries = llm_handler.generate_hierarchical_queries(original_query)
    hierarchical_docs = [
        doc
        for query in hierarchical_queries
        for doc in retrieve_documents([query], vector_store)
    ]
    hierarchical_fusion_response = llm_handler.generate_final_response(
        original_query, hierarchical_docs
    )

    # Cross-Encoder Reranking
    reranked_docs = cross_encoder_reranking(fusion_queries, fusion_docs)
    cross_encoder_response = llm_handler.generate_final_response(
        original_query, reranked_docs
    )

    # Multi-Stage Fusion
    multi_stage_docs = multi_stage_fusion(original_query, vector_store, fusion_docs)
    multi_stage_response = llm_handler.generate_final_response(
        original_query, multi_stage_docs
    )

    # Evaluate responses
    evaluation = llm_handler.evaluate_responses(
        standard_response,
        fusion_response,
        hierarchical_fusion_response,
        cross_encoder_response,
        multi_stage_response,
    )

    cursor.execute(
        "INSERT INTO evaluations (query, generated_queries, standard_response, fusion_response, hierarchical_fusion_response, cross_encoder_reranking_response, multi_stage_fusion_response, evaluation) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            original_query,
            " | ".join(fusion_queries + hierarchical_queries),
            standard_response,
            fusion_response,
            hierarchical_fusion_response,
            cross_encoder_response,
            multi_stage_response,
            evaluation,
        ),
    )
    conn.commit()


def search_arxiv(query, max_results=5):
    """Search Arxiv and download related PDFs, storing results in the database."""
    base_url = "http://export.arxiv.org/api/query"
    params = {"search_query": query, "start": 0, "max_results": max_results}
    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        root = ET.fromstring(response.text)
        pdf_links = []
        for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
            for link in entry.findall("{http://www.w3.org/2005/Atom}link"):
                if link.attrib.get("title") == "pdf":
                    pdf_links.append(link.attrib.get("href"))

        downloaded_pdfs = []
        for pdf_url in pdf_links:
            filename = os.path.join(DATA_DIR, pdf_url.split("/")[-1])
            pdf_response = requests.get(pdf_url)
            if pdf_response.status_code == 200:
                with open(filename, "wb") as f:
                    f.write(pdf_response.content)
                cursor.execute(
                    "INSERT INTO arxiv_search (query, pdf_url, filename) VALUES (?, ?, ?)",
                    (query, pdf_url, filename),
                )
                conn.commit()
                downloaded_pdfs.append(filename)
        return downloaded_pdfs
    return []


def extract_text_from_pdfs(pdf_files):
    """Extract text from a list of PDF files and store them in the database."""
    extracted_text = []
    for pdf_file in pdf_files:
        with open(pdf_file, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = "".join(
                [page.extract_text() for page in reader.pages if page.extract_text()]
            )
            extracted_text.append(text)
            cursor.execute(
                "INSERT INTO documents (filename, extracted_text) VALUES (?, ?)",
                (pdf_file, text),
            )
    conn.commit()
    return extracted_text


def split_text_on_sentences(text, chunk_size=500, overlap=50):
    """Split text into chunks while ensuring sentence boundaries and store in database."""
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    split_text = " | ".join(chunks)
    cursor.execute(
        "UPDATE documents SET split_text = ? WHERE extracted_text = ?",
        (split_text, text),
    )
    conn.commit()
    return chunks


def retrieve_documents(queries, vector_store, top_k=config["top_k"]):
    """Retrieve relevant documents for each query."""
    results = []
    for query in queries:
        query_embedding = generate_embeddings(query)
        scores, indices = vector_store.index.search(np.array([query_embedding]), top_k)
        results.append([(docs[i], scores[0][j]) for j, i in enumerate(indices[0])])
    return results


# Example usage
llm = OpenAI()
llm_handler = LLMHandler(llm)
original_query = "RAG Fusion"
evaluate_rag_variants(original_query, vector_store, llm_handler)
