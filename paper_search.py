import requests
import xml.etree.ElementTree as ET
import os
import logging
import PyPDF2
import datetime

from config import appConfig
from ollama_utils import generate_embeddings
from nltk.tokenize import sent_tokenize

logger = logging.getLogger(__name__)

from database  import Database

class PaperSearch:
    def __init__(self, config=appConfig):
        self.config = config
        self.base_url = "http://export.arxiv.org/api/query"
        self.max_results = config["max_search_results"]
        self.data_dir = config["DATA_DIR"]
        self.chunk_size = config["CHUNK_SIZE"]
        self.chunk_overlap = config["CHUNK_OVERLAP"]
        self.db = Database(config["DB_NAME"])
        self.conn = self.db.conn
        self.cursor = self.db.cursor


    def search(self, query):
        """Search Arxiv and download related PDFs, storing results in the database."""
        params = {"search_query": query, "start": 0, "max_results": self.max_results}
        response = requests.get(self.base_url, params=params)
        if response.status_code == 200:
            pdf_links = self.get_pdf_links(response)
            downloaded_pdfs = []
            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir)
            for pdf_url in pdf_links:
                pdf_file = self.download_pdf(pdf_url, query)
                if pdf_file:
                    downloaded_pdfs.append(pdf_file)
            logger.info(f"Downloaded {len(downloaded_pdfs)} PDFs for query: {query}")
            return downloaded_pdfs
        return []

    def get_pdf_links(self, response):
        """Extract PDF links from an Arxiv API response."""
        root = ET.fromstring(response.text)
        pdf_links = []
        for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
            for link in entry.findall("{http://www.w3.org/2005/Atom}link"):
                if link.attrib.get("title") == "pdf":
                    pdf_links.append(link.attrib.get("href"))
        return pdf_links

    def download_pdf(self, pdf_url, query):
        """Download a PDF file from a URL and store it in the database."""
        # Check if the document already exists in the database
        self.cursor.execute("SELECT id FROM paper_search WHERE filename = ?", (pdf_url,))
        result = self.cursor.fetchone()
        if result:
            logger.info(f"PDF already exists in database: {pdf_url}")
            return None  
        filename = os.path.join(self.data_dir, pdf_url.split("/")[-1])
        pdf_response = requests.get(pdf_url)
        if pdf_response.status_code == 200:
            pdf_data = pdf_response.content
            with open(filename, "wb") as f:
                f.write(pdf_data)
            self.cursor.execute(
                "INSERT INTO paper_search (query, pdf_url, filename, pdf_data) VALUES (?, ?, ?, ?)",
                (query, pdf_url, filename, pdf_data),
            )
            self.conn.commit()
            self.extract_text_from_pdf(filename)
            logger.info(f"Downloaded PDF: {pdf_url}")
            return filename
        return None
    
    def insert_query(self, query):
        self.cursor.execute('''INSERT OR IGNORE INTO query (query, datetime) VALUES (?, ?)''',
                       (query, datetime.now().isoformat()))
        self.conn.commit()
        self.cursor.execute('''SELECT id FROM query WHERE query = ?''', (query,))
        return self.cursor.fetchone()[0]


    def extract_text_from_pdf(self, pdf_file):
        """Extract text from a PDF file and store them in the database."""
        extracted_text = []
        with open(pdf_file, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            paper_search_id = self.cursor.execute("SELECT id FROM paper_search WHERE filename = ?", (pdf_file,)).fetchone()[0]
            for page in reader.pages:
                text = page.extract_text()
                extracted_text.append(text)
                self.cursor.execute(
                    """INSERT INTO document_page (paper_search_id, page_number, text) 
                    VALUES (?, ?, ?)
                    RETURNING id""",
                    (paper_search_id, pdf_file, text),
                )
                page_id = self.cursor.lastrowid
                self.split_text_on_sentences(page_id, text)
            self.conn.commit()

            extracted_text.append(text)
            self.cursor.execute(
                "INSERT INTO document_text (paper_search_id, text) VALUES (?, ?)",
                (paper_search_id, text),
            )
        self.conn.commit()
        return extracted_text

    def split_text_on_sentences(self, page_id, text):
        """Split text into chunks while ensuring sentence boundaries and store in database."""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += " " + sentence
            else:
                chunk_text = current_chunk.strip()
                chunks.append(chunk_text)
                current_chunk = sentence
                print(chunk_text)
        if current_chunk:
            chunks.append(current_chunk.strip())
        logger.info(f"Inserting {len(chunks)} chunks for text length {len(text)}")
        self.cursor.executemany(
            "INSERT INTO document_page_split(document_page_id, text) VALUES (?, ?)",
            [(page_id, chunk_text) for chunk_text in chunks]
        )
        logger.info(f"Inserted {len(chunks)} chunks")
        self.conn.commit()
        return chunks

    def create_embeddings(self):
        """Create embeddings for the extracted text."""
        self.cursor.execute("SELECT id, text FROM document_text")
        rows = self.cursor.fetchall()
        for row in rows:
            document_text_id, text = row
            embeddings = generate_embeddings(text)
            self.cursor.execute(
                "UPDATE document_text SET embeddings = ? WHERE id = ?",
                (embeddings, document_text_id),
            )
        self.conn.commit()


# se = PaperSearch(appConfig)
# se.search("RAG Fusion")