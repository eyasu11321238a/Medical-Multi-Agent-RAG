"""
pdf_ingestion.py
----------------
PDF ingestion pipeline using:
  - pdfplumber  → text extraction per page
  - langchain-text-splitters → RecursiveCharacterTextSplitter

Each chunk is tagged with metadata:
  cancer_type, chapter, page_number, source_file, content_type (text)
"""

import os
import re
from pathlib import Path
from typing import List, Dict

import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE",    500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP",  50))

CANCER_TYPE_MAP = {
    "basal_cell": "Basal Cell Skin Cancer",
    "melanoma":   "Melanoma",
    "squamous":   "Squamous Cell Skin Cancer",
    "breast":     "Breast Cancer",
    "lung":       "Lung Cancer",
    "colorectal": "Colorectal Cancer",
    "colon":      "Colorectal Cancer",
    "prostate":   "Prostate Cancer",
    "leukemia":   "Leukemia",
}

CHAPTER_KEYWORDS = [
    "about", "testing", "diagnosis", "types of treatment",
    "treatment by risk", "follow-up", "resources", "systemic",
    "surgery", "radiation", "clinical trial",
]


def detect_cancer_type(filename: str) -> str:
    fl = filename.lower()
    for key, label in CANCER_TYPE_MAP.items():
        if key in fl:
            return label
    return "Unknown Cancer Type"


def detect_chapter(text: str) -> str:
    tl = text.lower()
    for kw in CHAPTER_KEYWORDS:
        if kw in tl:
            return kw.title()
    return "General"


def load_pdf_text(pdf_path: str) -> List[Document]:
    """Extract text from every page via pdfplumber."""
    path        = Path(pdf_path)
    cancer_type = detect_cancer_type(path.name)
    docs        = []

    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if not text or len(text.strip()) < 50:
                continue
            text    = re.sub(r"\s+", " ", text).strip()
            chapter = detect_chapter(text)
            docs.append(Document(
                page_content=text,
                metadata={
                    "source_file":  path.name,
                    "cancer_type":  cancer_type,
                    "chapter":      chapter,
                    "page_number":  page_num,
                    "total_pages":  total,
                    "content_type": "text",
                },
            ))
    return docs


def load_pdf(pdf_path: str) -> List[Document]:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    cancer_type = detect_cancer_type(path.name)
    print(f"\n  📄 {path.name} → [{cancer_type}]")
    docs = load_pdf_text(pdf_path)
    print(f"     ✅ {len(docs)} pages extracted")
    return docs


def load_all_pdfs(pdf_dir: str) -> List[Document]:
    pdf_dir_path = Path(pdf_dir)
    if not pdf_dir_path.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")
    pdf_files = list(pdf_dir_path.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"No PDFs found in: {pdf_dir}")
    print(f"\n📚 Found {len(pdf_files)} PDF(s) in {pdf_dir}")
    all_documents = []
    for pdf_file in sorted(pdf_files):
        try:
            all_documents.extend(load_pdf(str(pdf_file)))
        except Exception as e:
            print(f"  ⚠️  Error loading {pdf_file.name}: {e}")
    print(f"\n✅ Total pages loaded: {len(all_documents)}")
    return all_documents


def chunk_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    print(f"\n✂️  Chunking {len(documents)} pages...")
    chunks = splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_total"] = len(chunks)
    print(f"✅ {len(chunks)} chunks ready for indexing")
    return chunks


def get_cancer_type_summary(documents: List[Document]) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for doc in documents:
        ct = doc.metadata.get("cancer_type", "Unknown")
        summary[ct] = summary.get(ct, 0) + 1
    return summary


def ingest_pipeline(pdf_dir: str) -> List[Document]:
    """
    Text-only PDF ingestion:
      1. Extract text from all PDFs (pdfplumber)
      2. Chunk with RecursiveCharacterTextSplitter
      3. Return chunks ready for FAISS indexing
    """
    print("=" * 55)
    print("   MEDICAL PDF INGESTION PIPELINE")
    print("   Text extraction: pdfplumber")
    print("=" * 55)

    raw_docs = load_all_pdfs(pdf_dir)

    summary = get_cancer_type_summary(raw_docs)
    print("\n📊 Cancer Types Loaded:")
    for ct, count in summary.items():
        print(f"   • {ct}: {count} pages")

    chunks = chunk_documents(raw_docs)
    print("\n🎉 Ingestion complete!")
    print("=" * 55)
    return chunks