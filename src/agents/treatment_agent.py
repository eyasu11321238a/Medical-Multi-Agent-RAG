"""
treatment_agent.py
------------------
Treatment Agent powered by Groq (llama-3.1-8b-instant).
Provides detailed treatment recommendations based on NCCN guidelines.
"""

import os
from typing import Optional, List, Dict

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

from src.rag.vector_store import MedicalRetriever
from src.utils.helpers import log_agent_call, clean_llm_response

load_dotenv()

TREATMENT_SYSTEM_PROMPT = """You are a specialized Medical Treatment Agent with expertise 
in cancer treatment protocols from NCCN (National Comprehensive Cancer Network) guidelines.

Provide structured, accurate treatment information covering:
- Surgical options
- Radiation therapy
- Systemic therapies (targeted, immunotherapy, chemotherapy)
- Topical treatments (when applicable)
- Clinical trial options
- Follow-up care recommendations

Structure responses with:
1. **Primary Treatment Options**
2. **Surgery Details**
3. **Non-Surgical Options**
4. **Systemic Therapy**
5. **Follow-up Care**
6. **When to Seek Specialist**

Distinguish preferred vs. alternative treatments.
Never recommend personal treatment plans — always advise consulting specialists.
"""

TREATMENT_USER_PROMPT = """Based on NCCN medical guidelines, provide comprehensive 
treatment information for:

**Question:** {query}
**Cancer Type:** {cancer_type}
**Risk Level:** {risk_level}

**Medical Context:**
{context}

Provide a detailed, well-structured treatment guide using only the context above.
"""

TREATMENT_COMPARISON_PROMPT = """Compare treatment approaches for these cancer types 
based on NCCN guidelines:

**Question:** {query}

**Medical Context:**
{context}

Structure your comparison covering:
1. Primary treatment differences
2. Surgical approaches
3. Drug/therapy options
4. When systemic therapy is used
"""


class TreatmentAgent:
    """Handles cancer treatment questions: surgery, drugs, radiation, clinical trials."""

    def __init__(self, retriever: MedicalRetriever):
        self.retriever = retriever
        self.llm = ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            temperature=float(os.getenv("GROQ_TEMPERATURE", 0.0)),
            max_tokens=int(os.getenv("GROQ_MAX_TOKENS", 2000)),
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )
        self.name = "Treatment Agent"

    def run(
        self,
        query: str,
        cancer_type: Optional[str] = None,
        risk_level: Optional[str] = None,
        k: int = 6,
    ) -> str:
        log_agent_call(self.name, query, cancer_type)

        docs = self.retriever.retrieve(
            query=f"treatment options surgery therapy {query}",
            cancer_type=cancer_type,
            k=k,
        )
        if not docs:
            return (
                "I could not find treatment information for your query. "
                "Please ensure the relevant cancer type PDF has been loaded."
            )

        context = self.retriever.format_context(docs)
        messages = [
            SystemMessage(content=TREATMENT_SYSTEM_PROMPT),
            HumanMessage(content=TREATMENT_USER_PROMPT.format(
                query=query,
                cancer_type=cancer_type or "Not specified",
                risk_level=risk_level or "Not specified",
                context=context,
            )),
        ]

        print(f"   🧠 Calling Groq (llama-3.1-8b-instant) for treatment...")
        response = self.llm.invoke(messages)
        result = clean_llm_response(response.content)
        print(f"   ✅ Treatment Agent completed")
        return result

    def run_comparison(
        self,
        query: str,
        cancer_types: List[str],
        k_per_type: int = 3,
    ) -> str:
        log_agent_call(f"{self.name} [Comparison]", query, str(cancer_types))

        multi_docs = self.retriever.retrieve_multi_cancer(
            query=f"treatment options surgery therapy {query}",
            cancer_types=cancer_types,
            k_per_type=k_per_type,
        )
        all_docs = [doc for docs in multi_docs.values() for doc in docs]
        context = self.retriever.format_context(all_docs)

        messages = [
            SystemMessage(content=TREATMENT_SYSTEM_PROMPT),
            HumanMessage(content=TREATMENT_COMPARISON_PROMPT.format(
                query=query, context=context
            )),
        ]
        print(f"   🧠 Calling Groq for treatment comparison...")
        response = self.llm.invoke(messages)
        return clean_llm_response(response.content)
