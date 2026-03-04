"""
app.py
------
Streamlit UI for the Medical Research Assistant.

Tech stack:
  - Groq (llama-3.1-8b-instant)  → all LLM agents
  - SentenceTransformers          → local embeddings (all-MiniLM-L6-v2)
  - FAISS                         → vector store
  - LangGraph                     → multi-agent orchestration

Run: streamlit run app.py
"""

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
from src.utils.tracing import setup_langsmith, get_run_url, is_tracing_enabled
sys.path.insert(0, str(Path(__file__).parent))

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Medical Research Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f, #2d6a9f);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    .tech-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: bold;
        margin: 2px;
    }
    .badge-groq  { background:#e3f2fd; color:#1565c0; }
    .badge-faiss { background:#fff3e0; color:#e65100; }
    .badge-lg    { background:#fce4ec; color:#880e4f; }
    .status-ok   { color: #2e7d32; font-weight: bold; }
    .status-warn { color: #e65100; font-weight: bold; }
    .disclaimer-box {
        background: #fff8e1;
        border-left: 4px solid #ffc107;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        font-size: 0.85rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────
def init_session():
    defaults = {
        "messages":      [],
        "graph":         None,
        "retriever":     None,
        "system_ready":  False,
        "pdf_count":     0,
        "pending_query": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()
setup_langsmith()


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ System Setup")

    # ── API Keys ──────────────────────────────
    st.markdown("### 🔑 API Keys")

    groq_key = st.text_input(
        "Groq API Key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        help="Get a free key at console.groq.com",
    )
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key

    if groq_key and groq_key != "your_groq_api_key_here":
        st.markdown('<span class="status-ok">✅ Groq ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-warn">⚠️ Groq key required</span>', unsafe_allow_html=True)

    st.markdown("---")

    # ── PDF Files ─────────────────────────────
    st.markdown("### 📁 PDF Guidelines")
    pdf_dir   = Path("data/raw_pdfs")
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = list(pdf_dir.glob("*.pdf"))

    if pdf_files:
        st.success(f"✅ {len(pdf_files)} PDF(s) ready")
        for f in pdf_files:
            st.caption(f"• {f.name}")
    else:
        st.warning("No PDFs found in data/raw_pdfs/")
        st.caption("Download from [NCCN Guidelines](https://www.nccn.org/patientguidelines)")

    st.markdown("---")

    # ── Options ───────────────────────────────
    st.markdown("### 🔧 Options")
    rebuild_index = st.checkbox("🔄 Rebuild FAISS index", value=False)

    st.markdown("---")

    # ── Initialize ────────────────────────────
    if st.button("🚀 Initialize System", type="primary", use_container_width=True):
        errors = []
        if not groq_key or groq_key == "your_groq_api_key_here":
            errors.append("Groq API key is required.")
        if not pdf_files:
            errors.append("No PDFs found in data/raw_pdfs/")

        if errors:
            for e in errors:
                st.error(e)
        else:
            progress = st.progress(0, text="Starting...")
            status   = st.empty()

            try:
                from src.rag.pdf_ingestion import ingest_pipeline
                from src.rag.vector_store import get_or_build_vector_store, MedicalRetriever
                from src.graph.medical_graph import build_medical_graph

                faiss_exists = (
                    Path(os.getenv("FAISS_INDEX_PATH", "data/faiss_index")) / "index.faiss"
                ).exists()

                if rebuild_index or not faiss_exists:
                    status.info("📄 Ingesting PDFs...")
                    progress.progress(20, text="Loading PDFs...")
                    chunks = ingest_pipeline("data/raw_pdfs")

                    progress.progress(60, text="Building FAISS index...")
                    vector_store = get_or_build_vector_store(chunks)
                else:
                    status.info("📂 Loading existing FAISS index...")
                    progress.progress(40, text="Loading FAISS index...")
                    vector_store = get_or_build_vector_store()

                progress.progress(80, text="Building LangGraph...")
                retriever = MedicalRetriever(vector_store)
                graph     = build_medical_graph(retriever)

                st.session_state.graph        = graph
                st.session_state.retriever    = retriever
                st.session_state.system_ready = True
                st.session_state.pdf_count    = len(pdf_files)

                progress.progress(100, text="Done!")
                status.success("✅ System ready! Start asking questions.")

            except Exception as e:
                progress.empty()
                status.error(f"❌ Initialization failed: {e}")
                st.exception(e)

    st.markdown("---")

    # ── Model Info ────────────────────────────
    st.markdown("### 🤖 Model Info")
    st.caption("**LLM:** Groq — llama-3.1-8b-instant")
    st.caption("**Embeddings:** all-MiniLM-L6-v2 (local)")
    st.caption("**Vector Store:** FAISS")
    st.caption("**Orchestration:** LangGraph")

    st.markdown("---")

    # ── LangSmith Tracing ─────────────────────
    st.markdown("### 🔭 LangSmith Tracing")
    ls_key = st.text_input(
        "LangSmith API Key",
        value=os.getenv("LANGCHAIN_API_KEY", ""),
        type="password",
        help="Get a free key at smith.langchain.com",
        key="ls_key_input",
    )
    if ls_key and ls_key != "your_langsmith_api_key_here":
        os.environ["LANGCHAIN_API_KEY"]    = ls_key
        os.environ["LANGCHAIN_TRACING_V2"] = "true"

    ls_project = st.text_input(
        "Project Name",
        value=os.getenv("LANGCHAIN_PROJECT", "medical-research-assistant"),
        key="ls_project_input",
    )
    if ls_project:
        os.environ["LANGCHAIN_PROJECT"] = ls_project

    if st.button("🔭 Enable Tracing", use_container_width=True, key="enable_tracing_btn"):
        enabled = setup_langsmith()
        if enabled:
            st.success("✅ LangSmith tracing enabled!")
            st.rerun()
        else:
            st.error("❌ Could not connect — check API key")

    if is_tracing_enabled():
        st.success("🟢 Tracing active")
        st.markdown(f"[📊 View Traces in LangSmith]({get_run_url()})")
    else:
        st.caption("⚫ Tracing inactive — add key above")

    st.markdown("---")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # ── Example Queries ───────────────────────
    st.markdown("### 💡 Try These")
    examples = [
        "What are signs of basal cell skin cancer?",
        "What surgery options exist for high-risk BCC?",
        "Compare melanoma vs basal cell cancer treatment",
        "What is Mohs surgery?",
        "What systemic therapies treat advanced skin cancer?",
        "What follow-up schedule is recommended after BCC treatment?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=f"ex_{ex[:25]}"):
            if st.session_state.system_ready:
                st.session_state.pending_query = ex
            else:
                st.warning("Initialize the system first.")


# ─────────────────────────────────────────────
# Main Panel
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h2 style="margin:0">🏥 Medical Research Assistant</h2>
    <p style="margin:0.4rem 0 0; opacity:0.85">
        Multi-Agent RAG &nbsp;|&nbsp; Groq · FAISS · LangGraph · LangSmith
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<span class="tech-badge badge-groq">🤖 Groq — Llama 3.1</span>'
    '<span class="tech-badge badge-faiss">🔍 FAISS — Vector Search</span>'
    '<span class="tech-badge badge-lg">🔗 LangGraph — Orchestration</span>',
    unsafe_allow_html=True,
)
st.markdown("")

if not st.session_state.system_ready:
    st.info("👈 Add your Groq API key, then click **Initialize System** in the sidebar.")
else:
    st.success(f"✅ System active — {st.session_state.pdf_count} PDF(s) indexed")

# Chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle example query click
pending    = st.session_state.pop("pending_query", None)
user_input = st.chat_input(
    "Ask about cancer diagnosis, treatment, symptoms...",
    disabled=not st.session_state.system_ready,
) or pending

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        if not st.session_state.system_ready or st.session_state.graph is None:
            err = (
                "⚠️ System is not initialized. "
                "Enter your Groq API key and click **🚀 Initialize System** first."
            )
            st.warning(err)
            st.session_state.messages.append({"role": "assistant", "content": err})
        else:
            with st.spinner("🤖 Agents analyzing your query..."):
                try:
                    from src.graph.medical_graph import run_graph

                    history = [
                        f"{m['role'].title()}: {m['content']}"
                        for m in st.session_state.messages[:-1]
                    ]
                    response = run_graph(
                        graph=st.session_state.graph,
                        query=user_input,
                        conversation_history=history,
                    )
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

                except Exception as e:
                    err = f"❌ Error: {str(e)}"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})

# Disclaimer
st.markdown("""
<div class="disclaimer-box">
⚠️ <strong>Medical Disclaimer:</strong> This tool provides information from NCCN patient
guidelines for <em>educational purposes only</em>. It is not a substitute for professional
medical advice. Always consult a qualified healthcare provider for personal medical decisions.
</div>
""", unsafe_allow_html=True)