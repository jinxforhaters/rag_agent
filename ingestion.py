from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer
import numpy as np
import re
from semantic_text_splitter import TextSplitter
from dotenv import load_dotenv


# ==========
# CONFIG
# ==========

@dataclass
class DataPipelineConfig:
    pdf_path: str
    url: str
    chunk_size: int = 800
    chunk_overlap: int = 100
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    collection_name: str = "agentic_rag_collection"


# ==========
# PIPELINE
# ==========

class DataIngestionPipeline:
    """
    End-to-end pipeline:
        PDF + URL -> cleaned docs -> chunks -> embeddings -> Qdrant vectorstore
    """

    def __init__(self, config: DataPipelineConfig):
        self.config = config

        self.embedder = HuggingFaceEmbeddings(
            model_name=self.config.embed_model_name
        )

        self.qdrant_client = QdrantClient(path="qdrant_local")

    # -------------
    # 1. LOAD STAGE
    # -------------

    def load_pdf(self) -> List[Document]:
        """Load PDF pages as LangChain Documents."""
        pdf_path = self.config.pdf_path
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found at: {pdf_path}")

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        for idx, d in enumerate(docs):
            d.metadata["source_type"] = "pdf"
            d.metadata["source"] = pdf_path
            d.metadata["page"] = idx           
            d.metadata["pages"] = [idx]   


        return docs

    def load_web(self) -> List[Document]:
        """Load web page text as LangChain Documents."""
        loader = WebBaseLoader(self.config.url)
        docs = loader.load()

        for d in docs:
            d.metadata["source_type"] = "web"
            d.metadata["source"] = self.config.url
            d.metadata["pages"] = [] 

        return docs

    # --------------------
    # 2. NORMALIZATION STAGE
    # --------------------

    @staticmethod
    def _simple_clean(text: str) -> str:
        """
        Basic cleanup:
        - remove non-breaking spaces
        - collapse multiple spaces/newlines
        NOTE: HTML boilerplate is mostly handled by WebBaseLoader.
        """
        text = text.replace("\xa0", " ")
        text = " ".join(text.split())
        return text
    
    def clean_html_artifacts(self, text: str) -> str:
        text = re.sub(r"\\b(read more|click here|subscribe|newsletter)\\b", "", text, flags=re.I)
        text = re.sub(r"http[s]?://\\S+", "", text)  # remove raw URLs
        text = re.sub(r"\\s+", " ", text)
        return text.strip()

    def normalize_docs(self, docs: List[Document]) -> List[Document]:
        """
        Normalize document texts and remove exact duplicates.
        """
        seen = set()
        cleaned_docs: List[Document] = []

        for d in docs:
            cleaned = self._simple_clean(d.page_content)
            cleaned = self.clean_html_artifacts(cleaned)
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                d.page_content = cleaned
                cleaned_docs.append(d)

        return cleaned_docs

    # -------------
    # 3. CHUNKING
    # -------------

    
    def semantic_text_split(self, docs: List[Document], max_tokens: int = 300):
        """
        Best-in-class semantic chunking using Karpathy's semantic-text-splitter.
        Meaning-preserving, topic-coherent chunks.
        """

        splitter = TextSplitter.from_tiktoken_model(
            "gpt-4o-mini",    
            capacity=128000   
        )

        chunks = []
        global_id = 0

        for doc in docs:
            text = doc.page_content
            page = doc.metadata.get("page")

            chunk_texts = splitter.chunks(text)


            for idx, chunk in enumerate(chunk_texts):
                chunks.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            **doc.metadata,
                            "chunk_id": global_id,
                            "chunk_index_in_source": idx,
                            "text_length": len(chunk),
                            "pages": [page],   # <---- FIX: correct page tracking
                        }
                    )
                )
                global_id += 1

        return chunks
    

    # -------------
    # 4. EMBEDDING + VECTOR DB
    # -------------

    def build_vectorstore(self, chunks):
        load_dotenv()

        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api = os.getenv("QDRANT_API_KEY")

        if not qdrant_url or not qdrant_api:
            raise ValueError("QDRANT_URL or QDRANT_API_KEY missing in .env")

        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api
        )

        dim = len(self.embedder.embed_query("dimension_probe"))

        client.recreate_collection(
            collection_name=self.config.collection_name,
            vectors_config=VectorParams(
                size=dim,
                distance=Distance.COSINE,
            )
        )

        vs = Qdrant(
            client=client,
            collection_name=self.config.collection_name,
            embeddings=self.embedder,
        )

        vs.add_documents(chunks)

        return vs
    # -------------
    # 5. END-TO-END RUN
    # -------------

    def run(self) -> Tuple[Qdrant, List[Document]]:
        """
        Flexible ingestion:
        - If PDF only → ingest PDF
        - If URL only → ingest URL
        - If both → ingest both
        """

        all_docs = []

        # -----------------------------
        # Load PDF if provided
        # -----------------------------
        if self.config.pdf_path and os.path.exists(self.config.pdf_path):
            print("🔹 Loading PDF...")
            pdf_docs = self.load_pdf()
            print(f"   Loaded {len(pdf_docs)} PDF pages")
            all_docs.extend(pdf_docs)
        else:
            print("⚠️ No PDF provided or file not found. Skipping PDF ingestion.")

        # -----------------------------
        # Load URL if provided
        # -----------------------------
        if self.config.url:
            print("🔹 Loading Web URL...")
            try:
                web_docs = self.load_web()
                print(f"   Loaded {len(web_docs)} web documents")
                all_docs.extend(web_docs)
            except Exception as e:
                print(f"⚠️ Failed to load URL: {e}")
        else:
            print("⚠️ No URL provided. Skipping URL ingestion.")

        # -----------------------------
        # Validate
        # -----------------------------
        if not all_docs:
            raise ValueError("❌ No valid sources provided. Please input a PDF or URL.")

        # -----------------------------
        # Normalize
        # -----------------------------
        print("🔹 Normalizing texts...")
        all_docs = self.normalize_docs(all_docs)

        # -----------------------------
        # Semantic Chunking
        # -----------------------------
        print("🔹 Chunking documents (semantic)...")
        chunks = self.semantic_text_split(all_docs, max_tokens=300)
        print(f"   Total chunks created: {len(chunks)}")

        # -----------------------------
        # Build vectorstore
        # -----------------------------
        print("🔹 Building vectorstore...")
        vs = self.build_vectorstore(chunks)
        print("✅ Vectorstore ready.")

        return vs, chunks


