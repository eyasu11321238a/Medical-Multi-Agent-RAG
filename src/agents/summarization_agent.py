"""
summarization_agent.py — Structured JSON output with citation validation.
"""

import os
from typing import Optional, List, Tuple

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from dotenv import load_dotenv

from src.rag.vector_store import MedicalRetriever
from src.utils.helpers import log_agent_call, clean_llm_response
from src.utils.structured_output import SummarizationResponse, MergedResponse
from src.utils.json_parser import force_json_response
from src.utils.citation_validator import validate_claims, filter_to_valid_claims, format_validated_response, ValidationReport

load_dotenv()

SYSTEM_PROMPT = """You are a Medical Summarization Agent specializing in NCCN guidelines.
Every factual claim must be cited with source_file, page_number, chapter, and a short verbatim quote.
Do not make any uncited factual claims.
"""

QA_PROMPT = """Answer this question using ONLY the context below.

Question: {query}
Cancer Type: {cancer_type}

Context:
{context}

Return a SummarizationResponse JSON. Every claim must cite source_file, page_number, and quote.
"""

COMPARISON_PROMPT = """Compare these cancer types based on the context below.

Question: {query}
Cancer Types: {cancer_types}

Context:
{context}

Return a SummarizationResponse JSON with cited comparison_points for each cancer type.
"""

MERGER_SYSTEM = """You are a medical content synthesizer.
Combine the provided diagnosis and treatment sections into one clean response.

CRITICAL FORMATTING RULE:
Both sections contain structured citation blocks in this exact format:
  > 📄 **Source:** `filename.pdf`
  > 📖 **Page:** N
  > 🗂️ **Chapter:** ...
  > 💬 **Quote:** "..."

You MUST preserve ALL citation blocks exactly as-is. 
Do NOT replace them with "(Source 1)" or any shorthand.
Do NOT rewrite, summarize, or strip citation metadata.
Simply combine the two sections with clear headings and add a summary.
"""

MERGER_PROMPT = """Combine these two validated sections into one response.
PRESERVE all citation blocks (📄/📖/🗂️/💬 lines) exactly as they appear.

Original Query: {query}
Cancer Type: {cancer_type}

--- DIAGNOSIS SECTION ---
{diagnosis_text}

--- TREATMENT SECTION ---
{treatment_text}

Output the combined response with:
1. A "## Diagnosis" heading followed by the diagnosis section (citations intact)
2. A "## Treatment" heading followed by the treatment section (citations intact)  
3. A "## Summary" heading with a 2-3 sentence plain-language summary
"""


class SummarizationAgent:

    def __init__(self, retriever: MedicalRetriever):
        self.retriever = retriever
        self.llm = ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            temperature=0.0,
            max_tokens=int(os.getenv("GROQ_MAX_TOKENS", 2000)),
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )
        self.name = "Summarization/Q&A Agent"

    def run(
        self,
        query: str,
        cancer_type: Optional[str] = None,
        k: int = 5,
    ) -> Tuple[Optional[SummarizationResponse], ValidationReport, List[Document]]:
        log_agent_call(self.name, query, cancer_type)

        docs = self.retriever.retrieve(query=query, cancer_type=cancer_type, k=k)
        if not docs:
            return None, _empty_report(), []

        context = self.retriever.format_context(docs)

        parsed, success, error = force_json_response(
            llm=self.llm,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=QA_PROMPT.format(
                query=query,
                cancer_type=cancer_type or "Not specified",
                context=context,
            ),
            output_model=SummarizationResponse,
            query=query, context=context,
        )

        if not success or parsed is None:
            print(f"   ❌ Summarization JSON failed: {error}")
            return None, _empty_report(), docs

        report = validate_claims(parsed.all_claims(), docs)
        print(f"   ✅ Summarization done — {report.valid_claims}/{report.total_claims} claims verified")
        return parsed, report, docs

    def run_as_text(self, query: str, cancer_type: Optional[str] = None) -> str:
        parsed, report, docs = self.run(query, cancer_type)
        if parsed is None:
            return "Could not retrieve information."
        valid_claims = filter_to_valid_claims(parsed.all_claims(), report)
        return format_validated_response(valid_claims, report, parsed.summary)

    def run_comparison(
        self,
        query: str,
        cancer_types: List[str],
        k_per_type: int = 3,
    ) -> Tuple[Optional[SummarizationResponse], ValidationReport, List[Document]]:
        log_agent_call(f"{self.name} [Comparison]", query, str(cancer_types))
        multi_docs = self.retriever.retrieve_multi_cancer(
            query=query, cancer_types=cancer_types, k_per_type=k_per_type
        )
        all_docs = [doc for docs in multi_docs.values() for doc in docs]
        context = self.retriever.format_context(all_docs)

        parsed, success, error = force_json_response(
            llm=self.llm,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=COMPARISON_PROMPT.format(
                query=query,
                cancer_types=", ".join(cancer_types),
                context=context,
            ),
            output_model=SummarizationResponse,
            query=query, context=context,
        )
        if not success or parsed is None:
            return None, _empty_report(), all_docs
        report = validate_claims(parsed.all_claims(), all_docs)
        return parsed, report, all_docs

    def merge_structured_outputs(
        self,
        query: str,
        cancer_type: Optional[str],
        intent: str,
        diagnosis_text: str,
        treatment_text: str,
    ) -> str:
        """Plain-text merge of already-validated diagnosis + treatment outputs."""
        messages = [
            SystemMessage(content=MERGER_SYSTEM),
            HumanMessage(content=MERGER_PROMPT.format(
                query=query,
                cancer_type=cancer_type or "Not specified",
                intent=intent,
                diagnosis_text=diagnosis_text,
                treatment_text=treatment_text,
            )),
        ]
        response = self.llm.invoke(messages)
        return clean_llm_response(response.content)

    def summarize_cancer_type(self, cancer_type: str, k: int = 8) -> str:
        parsed, report, docs = self.run(
            query=f"overview diagnosis treatment risk {cancer_type}",
            cancer_type=cancer_type, k=k,
        )
        if parsed is None:
            return "Summary could not be generated."
        valid_claims = filter_to_valid_claims(parsed.all_claims(), report)
        return format_validated_response(valid_claims, report, parsed.summary, cancer_type)

    def explain_term(self, term: str, cancer_type: Optional[str] = None) -> str:
        parsed, report, docs = self.run(query=term, cancer_type=cancer_type, k=3)
        if parsed is None:
            return "Term explanation could not be retrieved."
        valid_claims = filter_to_valid_claims(parsed.all_claims(), report)
        return format_validated_response(valid_claims, report, parsed.summary)


def _empty_report() -> ValidationReport:
    return ValidationReport(
        total_claims=0, valid_claims=0, invalid_claims=0,
        validation_results=[], overall_valid=False,
        needs_regeneration=False,
        failure_summary="No documents retrieved",
    )