"""
FAISS Vector Store Manager for FinSight AI

Manages FAISS vector storage with BGE-base embeddings for SEC filing analysis.
Features: MMR retrieval, ticker metadata filtering, section tagging, smart caching,
financial table preservation during chunking.
"""

import hashlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# SEC filing section detection patterns (10-K and 10-Q)
SECTION_PATTERNS = {
    "Item 1A - Risk Factors": re.compile(r"item\s+1a\b", re.IGNORECASE),
    "Item 1 - Business": re.compile(r"item\s+1\b(?!\s*[ab])", re.IGNORECASE),
    "Item 7 - MD&A": re.compile(r"item\s+7\b", re.IGNORECASE),
    "Item 8 - Financial Statements": re.compile(r"item\s+8\b", re.IGNORECASE),
    "Item 2 - Properties": re.compile(r"item\s+2\b", re.IGNORECASE),
    "Item 3 - Legal Proceedings": re.compile(r"item\s+3\b", re.IGNORECASE),
}

# Regex to detect financial table rows: multiple $ amounts or numbers in one line
TABLE_ROW_PATTERN = re.compile(
    r"(?:(?:[A-Za-z\s,\-\(\)]+)\s+(?:\$?\s*[\d,]+\.?\d*\s*){2,})",
    re.MULTILINE,
)


def detect_section(text: str) -> str:
    """Detect which SEC filing section a chunk belongs to based on its content."""
    for section_name, pattern in SECTION_PATTERNS.items():
        if pattern.search(text):
            return section_name
    return "General"


def preserve_financial_tables(text: str) -> str:
    """
    Wrap financial table blocks with sentinel separators so RecursiveCharacterTextSplitter
    treats them as atomic units and won't split mid-row.
    """
    lines = text.split("\n")
    result = []
    in_table = False
    table_buffer = []

    for line in lines:
        is_table_row = bool(TABLE_ROW_PATTERN.match(line.strip())) and len(line.strip()) > 10
        if is_table_row:
            if not in_table:
                in_table = True
                table_buffer = []
            table_buffer.append(line)
        else:
            if in_table:
                # End of table block - wrap as atomic unit
                result.append("\n<<<TABLE>>>\n" + "\n".join(table_buffer) + "\n<<<TABLE>>>\n")
                table_buffer = []
                in_table = False
            result.append(line)

    if in_table and table_buffer:
        result.append("\n<<<TABLE>>>\n" + "\n".join(table_buffer) + "\n<<<TABLE>>>\n")

    return "\n".join(result)


class BGEEmbeddings(Embeddings):
    """BGE embeddings with instruction prefixes for improved retrieval quality."""

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        batch_size: int = 16,
        normalize: bool = True,
        query_instruction: str = "Search for SEC filing content related to this financial topic: ",
        embed_instruction: str = "This is a passage from an SEC financial filing: ",
    ) -> None:
        self._hf = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={
                "normalize_embeddings": normalize,
                "batch_size": batch_size,
            },
        )
        self._query_instruction = query_instruction
        self._embed_instruction = embed_instruction

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed = [f"{self._embed_instruction}{t}" for t in texts]
        return self._hf.embed_documents(prefixed)

    def embed_query(self, text: str) -> list[float]:
        return self._hf.embed_query(f"{self._query_instruction}{text}")

    def __call__(self, input_texts):
        if isinstance(input_texts, list):
            return self.embed_documents(input_texts)
        return self.embed_query(input_texts)


class FAISSVectorManager:
    """Manages per-ticker FAISS vector stores with smart caching."""

    def __init__(self, base_dir: str = "faiss_stores"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.embeddings_model = None
        # In-memory cache: avoids re-loading FAISS from disk on every Q&A call
        self._store_cache: dict[str, FAISS] = {}

    def get_embeddings_model(self) -> BGEEmbeddings:
        """Get or create cached BGE embeddings model. Detects MPS/CUDA/CPU automatically."""
        if self.embeddings_model is None:
            if torch.backends.mps.is_available():
                device, batch_size = "mps", 16
                logger.info("Using MPS for embeddings")
            elif torch.cuda.is_available():
                device, batch_size = "cuda", 32
                logger.info("Using CUDA for embeddings")
            else:
                device, batch_size = "cpu", 16
                logger.info("Using CPU for embeddings")

            self.embeddings_model = BGEEmbeddings(
                model_name=EMBEDDING_MODEL,
                device=device,
                batch_size=batch_size,
                normalize=True,
            )
        return self.embeddings_model

    def _get_store_path(self, ticker: str, form_type: str = "10-K") -> Path:
        safe_type = form_type.replace("-", "")  # "10K" or "10Q"
        return self.base_dir / ticker.upper() / f"{ticker.upper()}_{safe_type}_faiss"

    def _get_metadata_path(self, ticker: str, form_type: str = "10-K") -> Path:
        return self._get_store_path(ticker, form_type) / "metadata.json"

    def _compute_content_hash(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _save_metadata(self, ticker: str, accession_number: str, content_hash: str, chunk_count: int, form_type: str = "10-K") -> None:
        metadata = {
            "ticker": ticker.upper(),
            "form_type": form_type,
            "accession_number": accession_number,
            "content_hash": content_hash,
            "chunk_count": chunk_count,
            "created_at": datetime.now().isoformat(),
            "model_name": EMBEDDING_MODEL,
        }
        metadata_path = self._get_metadata_path(ticker, form_type)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata for {ticker} ({form_type}): {chunk_count} chunks")

    def _load_metadata(self, ticker: str, form_type: str = "10-K") -> dict[str, Any] | None:
        metadata_path = self._get_metadata_path(ticker, form_type)
        if metadata_path.exists():
            try:
                with open(metadata_path) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load metadata for {ticker} ({form_type}): {e}")
        return None

    def needs_rebuild(self, ticker: str, accession_number: str, form_type: str = "10-K") -> bool:
        """
        Check if FAISS store needs to be rebuilt - uses accession number only.
        SEC filings are immutable by accession number, so no content download required.
        Called BEFORE downloading filing text so we can skip the download on cache hits.
        """
        store_path = self._get_store_path(ticker, form_type)
        if not store_path.exists():
            return True
        metadata = self._load_metadata(ticker, form_type)
        if not metadata:
            return True
        if accession_number != metadata.get("accession_number", ""):
            logger.info(f"New filing detected for {ticker} ({form_type}) - rebuilding store")
            return True
        logger.info(f"FAISS store is current for {ticker} ({form_type}) - skipping text download")
        return False

    def load_store(self, ticker: str, form_type: str = "10-K") -> FAISS | None:
        """
        Load FAISS store from in-memory cache or disk.
        In-memory cache prevents re-reading disk on every Q&A call within the same session.
        """
        ticker = ticker.upper()
        cache_key = f"{ticker}:{form_type}"
        if cache_key in self._store_cache:
            return self._store_cache[cache_key]

        store_path = self._get_store_path(ticker, form_type)
        if not store_path.exists():
            return None

        embeddings_model = self.get_embeddings_model()
        try:
            vs = FAISS.load_local(
                str(store_path),
                embeddings_model,
                allow_dangerous_deserialization=True,
            )
            self._store_cache[cache_key] = vs
            logger.info(f"Loaded FAISS store into memory for {ticker} ({form_type})")
            return vs
        except Exception as e:
            logger.error(f"Failed to load store for {ticker} ({form_type}): {e}")
            return None

    def create_vector_store(self, content: str, ticker: str, accession_number: str, form_type: str = "10-K") -> FAISS | None:
        """
        Create or load FAISS vector store from SEC filing content.

        Each chunk is tagged with:
          - ticker: for metadata filtering (prevents cross-company contamination)
          - section: detected SEC filing section (Item 1, 1A, 7, 8, etc.)
          - chunk_index: position in the document

        Financial table blocks are preserved as atomic units (not split mid-row).
        10-K and 10-Q filings are stored in separate paths to prevent overwriting.
        """
        if not ticker or not accession_number:
            raise ValueError("ticker and accession_number are required")

        ticker = ticker.upper()
        store_path = self._get_store_path(ticker, form_type)

        # Cache hit - load from memory/disk without reprocessing content
        if not self.needs_rebuild(ticker, accession_number, form_type):
            vs = self.load_store(ticker, form_type)
            if vs:
                return vs
            logger.warning(f"Cache check passed but load failed for {ticker} ({form_type}) - rebuilding")

        logger.info(f"Processing {len(content):,} chars for {ticker}")

        # Preserve financial tables as atomic units before chunking
        processed_content = preserve_financial_tables(content)

        embeddings_model = self.get_embeddings_model()

        document = Document(
            page_content=processed_content,
            metadata={
                "ticker": ticker,
                "form_type": form_type,
                "type": f"sec_{form_type.lower().replace('-', '')}",
                "accession_number": accession_number,
                "filing_year": accession_number.split("-")[1] if "-" in accession_number else "unknown",
            },
        )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,      # larger chunks capture more table context
            chunk_overlap=150,
            length_function=len,
            separators=[
                "<<<TABLE>>>",    # table blocks as primary separator (atomic units)
                "\n\n",
                "\n",
                " ",
                "",
            ],
        )

        split_docs = text_splitter.split_documents([document])

        # Tag each chunk with section metadata for cited_sections in Q&A
        for i, doc in enumerate(split_docs):
            doc.metadata["section"] = detect_section(doc.page_content)
            doc.metadata["chunk_index"] = i
            # Remove table sentinels from final chunk content
            doc.page_content = doc.page_content.replace("<<<TABLE>>>", "").strip()

        logger.info(f"Created {len(split_docs)} chunks for {ticker}")

        try:
            vector_store = FAISS.from_documents(split_docs, embeddings_model)
            vector_store.save_local(str(store_path))

            content_hash = self._compute_content_hash(content)
            self._save_metadata(ticker, accession_number, content_hash, len(split_docs), form_type)

            # Populate in-memory cache with freshly built store
            cache_key = f"{ticker}:{form_type}"
            self._store_cache[cache_key] = vector_store

            return vector_store

        except Exception as e:
            logger.error(f"Failed to create vector store for {ticker} ({form_type}): {e}")
            return None

    def get_mmr_retriever(self, ticker: str, form_type: str = "10-K", k: int = 12, lambda_mult: float = 0.5):
        """
        Load FAISS store for ticker+form_type and return an MMR retriever with metadata filtering.

        Uses Maximum Marginal Relevance to reduce redundancy in retrieved chunks.
        lambda_mult controls relevance vs diversity: 1.0 = pure relevance, 0.0 = pure diversity.
        Callers set per-section values — financial tables need high relevance, market position needs diversity.
        Filters by ticker metadata as a safety net against cross-company contamination.
        10-K and 10-Q stores are separate paths - will not cross-contaminate.
        """
        ticker = ticker.upper()
        vector_store = self.load_store(ticker, form_type)
        if not vector_store:
            return None

        total_chunks = vector_store.index.ntotal
        actual_k = min(k, total_chunks)
        actual_fetch_k = min(max(actual_k * 5, 60), total_chunks)

        return vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": actual_k,
                "fetch_k": actual_fetch_k,
                "lambda_mult": lambda_mult,
                "filter": {"ticker": ticker},
            },
        )

    def list_stores(self) -> list[str]:
        """Returns list of 'TICKER:form_type' strings for all existing stores."""
        results = []
        for path in self.base_dir.glob("*/*_faiss"):
            name = path.name  # e.g. META_10K_faiss
            if "_10K_faiss" in name:
                results.append(name.replace("_10K_faiss", "") + ":10-K")
            elif "_10Q_faiss" in name:
                results.append(name.replace("_10Q_faiss", "") + ":10-Q")
        return sorted(results)

    def get_store_info(self, ticker: str, form_type: str = "10-K") -> dict[str, Any]:
        ticker = ticker.upper()
        store_path = self._get_store_path(ticker, form_type)
        metadata = self._load_metadata(ticker, form_type)
        info = {
            "ticker": ticker,
            "form_type": form_type,
            "exists": store_path.exists(),
            "store_path": str(store_path),
            "metadata": metadata,
        }
        if metadata:
            info["summary"] = {
                "chunks": metadata.get("chunk_count", 0),
                "accession": metadata.get("accession_number", "unknown"),
                "created": metadata.get("created_at", "unknown"),
                "model": metadata.get("model_name", "unknown"),
            }
        return info


# Global singleton
faiss_manager = FAISSVectorManager()