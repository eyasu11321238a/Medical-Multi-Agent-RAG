"""
diagnosis_agent.py — Structured JSON with mandatory cited claims.
"""
import os
from typing import Optional, List, Tuple

from langchain_groq import ChatGroq
from langchain_core.documents import Document
from dotenv import load_dotenv

from src.rag.vector_store import MedicalRetriever
from src.utils.helpers import log_agent_call
from src.utils.structured_output import DiagnosisResponse
from src.utils.json_parser import force_json_response
from src.utils.citation_validator import validate_claims, filter_to_valid_claims, format_validated_response, ValidationReport

load_dotenv()

SYSTEM_PROMPT = """You are a Medical Diagnosis Agent extracting diagnosis information
from NCCN cancer guidelines.

You MUST extract each finding as a separate claim in the appropriate array:
- symptoms: signs and symptoms of the cancer
- risk_factors: known risk factors  
- diagnostic_tests: tests used for diagnosis
- overview: general overview facts

Each claim MUST have a citation with source_file, page_number, and verbatim quote.
Use the [Source N] metadata headers in the context to get exact filename and page number.
"""

USER_PROMPT = """Extract all diagnosis information from the context below as structured JSON.

Question: {query}
Cancer Type: {cancer_type}

Context (each source shows filename and page number in the [Source N] header):
{context}

Populate symptoms, risk_factors, and overview arrays.
Each entry needs claim_id, statement, confidence, and citation (source_file, page_number, quote).
"""


class DiagnosisAgent:

    def __init__(self, retriever: MedicalRetriever):
        self.retriever = retriever
        self.llm = ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            temperature=0.0,
            max_tokens=int(os.getenv("GROQ_MAX_TOKENS", 2000)),
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )
        self.name = "Diagnosis Agent"

    def run(
        self,
        query: str,
        cancer_type: Optional[str] = None,
        k: int = 6,
    ) -> Tuple[Optional[DiagnosisResponse], ValidationReport, List[Document]]:
        log_agent_call(self.name, query, cancer_type)

        docs = self.retriever.retrieve(query=query, cancer_type=cancer_type, k=k)
        if not docs:
            return None, _empty_report("No documents retrieved"), []

        context = self.retriever.format_context(docs)

        parsed, success, error = force_json_response(
            llm=self.llm,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=USER_PROMPT.format(
                query=query,
                cancer_type=cancer_type or "Not specified",
                context=context,
            ),
            output_model=DiagnosisResponse,
            query=query,
            context=context,
        )

        if not success or parsed is None:
            print(f"   ❌ JSON failed: {error}")
            return None, _empty_report(error), docs

        all_claims = parsed.all_claims()
        print(f"   Claims extracted: {len(all_claims)}")
        report = validate_claims(all_claims, docs)
        print(f"   ✅ Done — {report.valid_claims}/{report.total_claims} claims verified")
        return parsed, report, docs

    def run_as_text(self, query: str, cancer_type: Optional[str] = None) -> str:
        parsed, report, _ = self.run(query, cancer_type)
        if parsed is None:
            return "Diagnosis information could not be retrieved."
        valid = filter_to_valid_claims(parsed.all_claims(), report)
        return format_validated_response(valid, report, parsed.summary, parsed.cancer_type)

    def identify_cancer_type(self, query: str) -> Optional[str]:
        from langchain_core.messages import HumanMessage
        r = self.llm.invoke([HumanMessage(
            content=f"What cancer type is mentioned? Return ONLY the name or 'Unknown'.\nQuery: {query}"
        )])
        t = r.content.strip()
        return None if t.lower() in ["unknown", "none", ""] else t


def _empty_report(msg: str = "") -> ValidationReport:
    return ValidationReport(
        total_claims=0, valid_claims=0, invalid_claims=0,
        validation_results=[], overall_valid=False,
        needs_regeneration=False, failure_summary=msg,
    )