"""
vector_store.py
---------------
Builds and manages the FAISS vector store using local
SentenceTransformers embeddings (no API key required).
"""

import os
import pickle
from pathlib import Path
from typing import List, Optional, Dict

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/faiss_index")
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
TOP_K            = int(os.getenv("TOP_K_RESULTS", 5))


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Initialize local SentenceTransformers embedding model.
    Downloads model on first run (~90MB), then cached locally.
    """
    print(f"   Loading embedding model: {EMBEDDING_MODEL}")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(chunks: List[Document]) -> FAISS:
    """Build a FAISS vector store from document chunks and save to disk."""
    print("\n🔨 Building FAISS vector store...")
    print(f"   Embedding model : {EMBEDDING_MODEL}")
    print(f"   Total chunks    : {len(chunks)}")

    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings)

    save_path = Path(FAISS_INDEX_PATH)
    save_path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(save_path))

    print(f"   ✅ FAISS index saved → {save_path}")
    return vector_store


def load_vector_store() -> Optional[FAISS]:
    """Load an existing FAISS vector store from disk."""
    save_path = Path(FAISS_INDEX_PATH)
    index_file = save_path / "index.faiss"

    if not index_file.exists():
        print("⚠️  No saved FAISS index found.")
        return None

    print(f"📂 Loading FAISS index from {save_path}...")
    embeddings = get_embeddings()
    vector_store = FAISS.load_local(
        folder_path=str(save_path),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )
    print("   ✅ FAISS index loaded successfully")
    return vector_store


def get_or_build_vector_store(chunks: Optional[List[Document]] = None) -> FAISS:
    """Load existing vector store or build a new one from chunks."""
    existing = load_vector_store()
    if existing:
        return existing
    if chunks is None:
        raise ValueError(
            "No saved FAISS index found and no chunks provided. "
            "Run the ingestion pipeline first."
        )
    return build_vector_store(chunks)


class MedicalRetriever:
    """
    Smart retriever supporting:
    - General search across all cancer types
    - Filtered search by specific cancer type
    - Multi-cancer comparison queries
    """

    def __init__(self, vector_store: FAISS):
        self.vector_store = vector_store
        self.top_k = TOP_K

    def retrieve(
        self,
        query: str,
        cancer_type: Optional[str] = None,
        k: Optional[int] = None,
    ) -> List[Document]:
        k = k or self.top_k
        if cancer_type:
            return self._filtered_retrieve(query, cancer_type, k)
        return self.vector_store.similarity_search(query, k=k)

    def _filtered_retrieve(self, query: str, cancer_type: str, k: int) -> List[Document]:
        candidates = self.vector_store.similarity_search(query, k=k * 4)
        filtered = [
            doc for doc in candidates
            if cancer_type.lower() in doc.metadata.get("cancer_type", "").lower()
        ]
        return filtered[:k] if filtered else candidates[:k]

    def retrieve_multi_cancer(
        self,
        query: str,
        cancer_types: List[str],
        k_per_type: int = 3,
    ) -> Dict[str, List[Document]]:
        return {ct: self._filtered_retrieve(query, ct, k_per_type) for ct in cancer_types}

    def format_context(self, documents: List[Document]) -> str:
        parts = []
        for i, doc in enumerate(documents, 1):
            meta = doc.metadata
            header = (
                f"[Source {i}] "
                f"{meta.get('cancer_type', 'Unknown')} | "
                f"Chapter: {meta.get('chapter', 'N/A')} | "
                f"Page: {meta.get('page_number', 'N/A')}"
            )
            parts.append(f"{header}\n{doc.page_content}")
        return "\n\n" + ("\n\n" + "─" * 60 + "\n\n").join(parts)