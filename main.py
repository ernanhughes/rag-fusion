import numpy as np
import nltk
#from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
import numpy as np

from config import appConfig
from database import Database
from faiss_db import FaissDB

import logging
nltk.download("punkt")

from ollama import chat
from ollama import ChatResponse

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='rag_fusion.log',  # Specify the filename
    filemode='w'  # Optional: 'w' for write (overwrite), 'a' for append
)

logger = logging.getLogger(__name__)
from collections import OrderedDict

from query_generator import QueryGenerator
from paper_search import PaperSearch
from faiss_db import FaissDB



vector_store = FaissDB()
docs = vector_store.load_documents_from_db(appConfig, 30)
embeddings = vector_store.get_embeddings(docs)
index = vector_store.build_index(embeddings)
vector_store.set_index(index)

qg = QueryGenerator(appConfig, vector_store)
qg.evaluate_rag_variants("RAG Fusion", vector_store)
