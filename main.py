import faiss
import numpy as np
import sqlite3
import nltk
import os
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

from config import appConfig

from faiss_db import FaissDB

# Initialize embeddings model
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")



class LLMHandler:
    """Handles all LLM calls in one place."""

    def __init__(self):
        self.llm = chat

    def generate_queries(self, original_query):
        prompt = f"Generate {appConfig.appConfig['num_queries']} search queries related to: {original_query}"
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

def cross_encoder_reranking(queries, retrieved_docs):
    """Rerank documents using a cross-encoder model."""
    scores = cross_encoder.predict([(query, doc) for query in queries for doc in retrieved_docs])
    reranked_docs = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in reranked_docs]


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





def retrieve_documents(queries, vector_store, top_k=appConfig["top_k"]):
    """Retrieve relevant documents for each query."""
    results = []
    for query in queries:
        query_embedding = vector_store.generate_embedding(query)
        scores, indices = vector_store.index.search(np.array([query_embedding]), top_k)
        results.append([(docs[i], scores[0][j]) for j, i in enumerate(indices[0])])
    return results


# Example usage
vector_store = FaissDB()
llm_handler = LLMHandler()
