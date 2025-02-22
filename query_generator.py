import re
from typing import List

from database import Database
from config import appConfig
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

from faiss_db import FaissDB, SearchResult
import numpy as np

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

import logging
from ollama import chat
from ollama import ChatResponse

logger = logging.getLogger(__name__)


class QueryGenerator:
    def __init__(self, config=appConfig, vector_store:FaissDB=None):
        self.config = config
        self.regex = r"\d+\.\s(.+)"  # Regex to extract numbered results
        self.db = Database()
        self.conn = self.db.conn
        self.cursor = self.db.cursor
        self.vector_store = vector_store

    def llm(self, prompt):
        logger.debug(f"Calling {appConfig['CHAT_MODEL']} with prompt: {prompt}")
        response: ChatResponse = chat(model=appConfig["CHAT_MODEL"], messages=[
            {
                'role': 'user',
                'content': prompt
            },
        ])
        logger.debug(f"Response {response.message.content}.")
        return response.message.content



    def store_query(self, original_query, prompt, generated_text):
        """Store the generated query in the SQLite database."""
        matches = re.findall(self.regex, generated_text)
        for query in matches:
            self.cursor.execute(
                "INSERT INTO queries (original_query, prompt, generated_query) VALUES (?, ?, ?)",
                (original_query, prompt, query),
            )
        self.conn.commit()

    def generate_queries(self, original_query):
        prompt = f"""Generate a list of {appConfig['num_queries']} search queries 
                    related to: {original_query} in numbered format
                    1. First query
                    2. Second query
                    3. Third query
                    """
        generated_text = self.llm(prompt)
        self.store_query(original_query, prompt, generated_text)
        matches = re.findall(self.regex, generated_text)
        return matches

    def generate_hierarchical_queries(self, original_query):
        # Step 1: Generate broad categories
        category_prompt = f"""Generate broad categories for: {original_query} 
                              Please return the categories in numbered format:
                              1. First Category
                              2. Second Category
                              3. Third Category
                           """
        generated_text = self.llm(category_prompt)
        self.store_query(original_query, category_prompt, generated_text)

        categories = re.findall(self.regex, generated_text)

        all_queries = []
        for category in categories:
            # Step 2: Generate specific queries for each category
            query_prompt = f"""Generate specific search queries for the category: {category}
                               Please return the queries in numbered format:
                               1. First query
                               2. Second query
                               3. Third query
                            """
            generated_text = self.llm(query_prompt)
            self.store_query(original_query, query_prompt, generated_text)

            queries = re.findall(self.regex, generated_text)
            all_queries.extend(queries)

        return all_queries

    def generate_final_response(self, original_query, ranked_docs):
        context = "\n".join(ranked_docs)
        prompt = f"Given the following context:\n{context}\nAnswer the query: {original_query}"
        generated_text = self.llm(prompt)
        self.store_query(original_query, prompt, generated_text)
        return generated_text
    
    def cross_encoder_reranking(self, queries, retrieved_docs):
        """Re-rank documents using a cross-encoder model."""
        scores = cross_encoder.predict([(query, doc) for query in queries for doc in retrieved_docs])
        reranked_docs = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in reranked_docs]

    def multi_stage_fusion(self, query, corpus):
        """Combine BM25 (sparse) and FAISS (dense) retrieval results before fusion."""
        tokenized_corpus = [doc.split() for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_scores = bm25.get_scores(query.split())

        # dense_search_results = self.vector_store.search(query)
        # dense_results = []
        # for query in dense_search_results:
        #     for doc in self.vector_store.search(query):
        #         standard_docs.append(doc.doc)


        combined_results = {
            doc: bm25_scores[i] + dense_results[i][1] for i, doc in enumerate(corpus)
        }

        return sorted(
            combined_results.keys(), key=lambda x: combined_results[x], reverse=True
        )


    def evaluate_responses(
        self,
        standard_response,
        fusion_response,
        hierarchical_fusion_response,
        multi_stage_response,
    ):
        evaluation_prompt = (
            f"Evaluate the following five responses and assign a grade based on accuracy, relevance, and comprehensiveness.\n\n"
            f"Standard RAG Response:\n{standard_response}\n\n"
            f"RAG-Fusion Response:\n{fusion_response}\n\n"
            f"Hierarchical RAG-Fusion Response:\n{hierarchical_fusion_response}\n\n"
            f"Multi-Stage Fusion Response:\n{multi_stage_response}\n\n"
            f"Provide a rating (1-10) and a brief justification for each response."
        )
        return self.llm(evaluation_prompt)

    def evaluate_rag_variants(self, original_query, vector_store):
        """Evaluate and compare multiple RAG approaches."""

        print(f"Evaluating RAG variants for query: {original_query}")
        # Standard RAG
        standard_queries = self.generate_queries(original_query)
        standard_docs = []
        for query in standard_queries:
            for doc in self.vector_store.search(query):
                standard_docs.append(doc.doc)
        standard_response = self.generate_final_response(
            original_query, standard_docs
        )

        # RAG-Fusion
        fusion_queries = self.generate_queries(original_query)
        fusion_docs = []
        fusion_docs_with_scores = []
        for query in fusion_queries:
            for doc in self.vector_store.search(query):
                fusion_docs.append(doc.doc)
                fusion_docs_with_scores.append((doc.doc, doc.score))
        fusion_response = self.generate_final_response(original_query, fusion_docs)

        # Hierarchical RAG-Fusion
        hierarchical_queries = self.generate_hierarchical_queries(original_query)
        hierarchical_docs = []
        for query in hierarchical_queries:
            for doc in self.vector_store.search(query):
                hierarchical_docs.append(doc.doc)
        hierarchical_fusion_response = self.generate_final_response(
            original_query, hierarchical_docs
        )

        # Cross-Encoder Reranking
        reranked_docs = self.cross_encoder_reranking(fusion_queries, fusion_docs)
        cross_encoder_response = self.generate_final_response(
            original_query, reranked_docs
        )

        # Multi-Stage Fusion
        multi_stage_docs = self.multi_stage_fusion(original_query, vector_store, fusion_docs_with_scores)
        multi_stage_response = self.generate_final_response(
            original_query, multi_stage_docs
        )

        # Evaluate responses
        evaluation = self.evaluate_responses(
            standard_response,
            fusion_response,
            hierarchical_fusion_response,
            multi_stage_response,
        )

        self.cursor.execute(
            "INSERT INTO evaluations (query, generated_queries, standard_response, fusion_response, hierarchical_fusion_response, multi_stage_fusion_response, evaluation) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                original_query,
                " | ".join(fusion_queries + hierarchical_queries),
                standard_response,
                fusion_response,
                hierarchical_fusion_response,
                multi_stage_response,
                evaluation,
            ),
        )
        self.conn.commit()

    def convert_to_text(self, documents: List[SearchResult]):
        res = []
        for doc in documents:
            res.append(doc.doc)
        return res


    def retrieve_documents(self, queries, vector_store, top_k=appConfig["top_k"]):
        """Retrieve relevant documents for each query."""
        results = []
        for query in queries:
            search_results = vector_store.search(query, top_k)
            for search_result in search_results:
                results.append(search_result.doc)
        return results


    # Reciprocal Rank Fusion algorithm
    # search_results_dict => {doc: score}
    def reciprocal_rank_fusion(self, search_results_dict, k=60):
        fused_scores = {}
        print("Initial individual search result ranks:")
        for query, doc_scores in search_results_dict.items():
            print(f"For query '{query}': {doc_scores}")
            
        for query, doc_scores in search_results_dict.items():
            for rank, (doc, score) in enumerate(sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)):
                if doc not in fused_scores:
                    fused_scores[doc] = 0
                previous_score = fused_scores[doc]
                fused_scores[doc] += 1 / (rank + k)
                print(f"Updating score for {doc} from {previous_score} to {fused_scores[doc]} based on rank {rank} in query '{query}'")

        reranked_results = {doc: score for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)}
        print("Final reranked results:", reranked_results)
        return reranked_results
