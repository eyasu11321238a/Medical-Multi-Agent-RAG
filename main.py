"""
main.py
-------
Entry point for the Medical Research Assistant.
Handles: PDF ingestion → Vector store → LangGraph → Streamlit UI
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

from src.rag.pdf_ingestion import ingest_pipeline
from src.rag.vector_store import get_or_build_vector_store, MedicalRetriever
from src.graph.medical_graph import build_medical_graph, run_graph

PDF_DIR       = "data/raw_pdfs"
FAISS_PATH    = os.getenv("FAISS_INDEX_PATH", "data/faiss_index")
REBUILD_INDEX = os.getenv("REBUILD_INDEX", "false").lower() == "true"


def check_environment():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        print("❌ ERROR: GROQ_API_KEY not set in .env file")
        print("   Get a free key at: https://console.groq.com")
        sys.exit(1)
    print("✅ Groq API key detected")


def check_pdfs() -> bool:
    pdf_path = Path(PDF_DIR)
    pdf_path.mkdir(parents=True, exist_ok=True)
    pdfs = list(pdf_path.glob("*.pdf"))
    if not pdfs:
        print(f"\n⚠️  No PDFs found in {PDF_DIR}/")
        print("   Download from: https://www.nccn.org/patientguidelines")
        return False
    print(f"✅ Found {len(pdfs)} PDF(s):")
    for pdf in pdfs:
        print(f"   • {pdf.name}")
    return True


def setup_vector_store() -> MedicalRetriever:
    faiss_exists = (Path(FAISS_PATH) / "index.faiss").exists()
    if REBUILD_INDEX or not faiss_exists:
        print("\n🔨 Building vector store from PDFs...")
        chunks = ingest_pipeline(PDF_DIR)
        vector_store = get_or_build_vector_store(chunks)
    else:
        print("\n📂 Loading existing FAISS index...")
        vector_store = get_or_build_vector_store()
    return MedicalRetriever(vector_store)


def interactive_loop(graph):
    conversation_history = []
    print("\n" + "=" * 55)
    print("   🏥 MEDICAL RESEARCH ASSISTANT — Ready!")
    print("=" * 55)
    print("Type your question. Commands: 'quit' | 'clear'\n")

    while True:
        try:
            user_input = input("\n🧑 You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["quit", "exit", "q"]:
                print("\n👋 Goodbye!")
                break
            if user_input.lower() == "clear":
                conversation_history = []
                print("🔄 History cleared.")
                continue

            response = run_graph(graph, user_input, conversation_history)
            conversation_history.append(f"User: {user_input}")
            conversation_history.append(f"Assistant: {response[:300]}...")
            print(f"\n🤖 Assistant:\n{response}")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


def main():
    print("\n" + "=" * 55)
    print("   🏥 MEDICAL RESEARCH ASSISTANT")
    print("   LangGraph + Groq (Llama 3.1) + FAISS + SentenceTransformers")
    print("=" * 55 + "\n")

    check_environment()

    has_pdfs = check_pdfs()
    if not has_pdfs:
        sys.exit(1)

    retriever = setup_vector_store()
    graph = build_medical_graph(retriever)
    interactive_loop(graph)


if __name__ == "__main__":
    main()
