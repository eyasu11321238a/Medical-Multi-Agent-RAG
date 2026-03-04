"""
verifier_agent.py
-----------------
Verifier Agent: Cross-checks the generated response against the
retrieved source chunks. Flags any claims not supported by the
NCCN guidelines context and returns a cleaned, verified response.

Anti-hallucination layer #1.
"""

import os
import re
from typing import Optional, List, Tuple

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from dotenv import load_dotenv

from src.rag.vector_store import MedicalRetriever
from src.utils.helpers import clean_llm_response, log_agent_call

load_dotenv()

VERIFIER_SYSTEM_PROMPT = """You are a strict Medical Fact-Verification Agent.
Your job is to verify that every claim in a generated medical response is 
directly supported by the provided NCCN guideline source documents.

Your verification process:
1. Read each factual claim in the response
2. Check whether the source context explicitly supports that claim
3. Mark claims as SUPPORTED, UNSUPPORTED, or INFERRED

Rules:
- SUPPORTED: The claim is clearly stated or directly implied in the source
- UNSUPPORTED: The claim cannot be found in the source context at all
- INFERRED: The claim is a reasonable inference but not explicitly stated

CRITICAL FORMATTING RULE:
The response contains structured citation blocks in this format:
  > 📄 **Source:** `filename.pdf`
  > 📖 **Page:** N
  > 🗂️ **Chapter:** ...
  > 💬 **Quote:** "..."

You MUST preserve ALL citation blocks exactly as-is in the verified response.
Do NOT rewrite citation blocks. Do NOT replace them with "(Source 1)", "(Source 2)" etc.
Only remove the claim text and its citation block together if UNSUPPORTED.

Output Format:
---VERIFICATION REPORT---
Overall Confidence: [HIGH / MEDIUM / LOW]
Unsupported Claims Found: [number]

Claim Analysis:
[List each major factual claim and its SUPPORTED/UNSUPPORTED/INFERRED status]

---VERIFIED RESPONSE---
[Copy the original response, keeping all citation blocks intact.
Remove entire claim+citation blocks only if UNSUPPORTED.
Do NOT summarize, rewrite, or strip citation metadata.]
"""

VERIFIER_USER_PROMPT = """Please verify the following medical response against 
the source context. PRESERVE all citation blocks (📄 Source, 📖 Page, 💬 Quote lines).

**Original Response to Verify:**
{response}

**Source Context (NCCN Guidelines):**
{context}

Verify each claim. Keep citation blocks exactly as formatted. Remove only UNSUPPORTED claim+citation pairs.
"""

QUICK_CHECK_PROMPT = """You are a medical fact checker. Given this response and 
source context, answer only:
1. Does the response contain any claims NOT found in the source? (Yes/No)
2. Confidence level of the response: (HIGH/MEDIUM/LOW)
3. Any specific unsupported claim to flag (one sentence max, or "None")

Response: {response}

Source: {context}

Answer in exactly this format:
HALLUCINATION_DETECTED: [Yes/No]
CONFIDENCE: [HIGH/MEDIUM/LOW]
FLAG: [specific claim or None]
"""


class VerifierAgent:
    """
    Verifies generated responses against retrieved source documents.
    
    Two modes:
    - full_verify(): Detailed claim-by-claim verification + rewrite
    - quick_check(): Fast pass/fail check with confidence score
    """

    def __init__(self, retriever: MedicalRetriever):
        self.retriever = retriever
        self.llm = ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            temperature=0.0,   # Always 0 for verification — no creativity
            max_tokens=int(os.getenv("GROQ_MAX_TOKENS", 2000)),
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )
        self.name = "Verifier Agent"

    def full_verify(
        self,
        response: str,
        query: str,
        cancer_type: Optional[str] = None,
        source_docs: Optional[List[Document]] = None,
    ) -> Tuple[str, dict]:
        """
        Full verification: checks each claim and rewrites the response
        to remove or soften unsupported statements.

        Returns:
            (verified_response, verification_metadata)
        """
        log_agent_call(self.name, query, cancer_type)

        # Get source docs if not provided
        if not source_docs:
            source_docs = self.retriever.retrieve(
                query=query,
                cancer_type=cancer_type,
                k=8,  # Get more docs for thorough verification
            )

        context = self.retriever.format_context(source_docs)

        messages = [
            SystemMessage(content=VERIFIER_SYSTEM_PROMPT),
            HumanMessage(content=VERIFIER_USER_PROMPT.format(
                response=response,
                context=context,
            )),
        ]

        print(f"   🔍 Verifier: checking response against {len(source_docs)} source chunks...")
        raw = self.llm.invoke(messages)
        raw_text = clean_llm_response(raw.content)

        # Parse the verification report
        metadata = self._parse_verification_report(raw_text)

        # Extract verified response section
        verified_response = self._extract_verified_response(raw_text, response)

        print(f"   ✅ Verifier done | Confidence: {metadata.get('confidence', 'N/A')} "
              f"| Unsupported claims: {metadata.get('unsupported_count', '?')}")

        return verified_response, metadata

    def quick_check(
        self,
        response: str,
        query: str,
        cancer_type: Optional[str] = None,
    ) -> dict:
        """
        Fast hallucination check. Returns a simple pass/fail dict.
        Used when speed matters more than depth.
        """
        source_docs = self.retriever.retrieve(
            query=query,
            cancer_type=cancer_type,
            k=5,
        )
        context = self.retriever.format_context(source_docs)

        messages = [
            HumanMessage(content=QUICK_CHECK_PROMPT.format(
                response=response[:1500],  # Truncate for speed
                context=context[:2000],
            )),
        ]

        raw = self.llm.invoke(messages)
        return self._parse_quick_check(raw.content)

    def _parse_verification_report(self, text: str) -> dict:
        """Extract metadata from verification report."""
        metadata = {
            "confidence": "UNKNOWN",
            "unsupported_count": 0,
            "has_report": False,
        }

        if "---VERIFICATION REPORT---" in text:
            metadata["has_report"] = True

        # Extract confidence level
        conf_match = re.search(r"Overall Confidence:\s*(HIGH|MEDIUM|LOW)", text, re.IGNORECASE)
        if conf_match:
            metadata["confidence"] = conf_match.group(1).upper()

        # Extract unsupported claim count
        unsup_match = re.search(r"Unsupported Claims Found:\s*(\d+)", text, re.IGNORECASE)
        if unsup_match:
            metadata["unsupported_count"] = int(unsup_match.group(1))

        return metadata

    def _extract_verified_response(self, full_text: str, original: str) -> str:
        """Pull out just the verified response section."""
        if "---VERIFIED RESPONSE---" in full_text:
            parts = full_text.split("---VERIFIED RESPONSE---")
            if len(parts) > 1:
                return clean_llm_response(parts[1])
        # If parsing fails, return original with a note
        return original

    def _parse_quick_check(self, text: str) -> dict:
        """Parse quick check output into structured dict."""
        result = {
            "hallucination_detected": False,
            "confidence": "MEDIUM",
            "flag": "None",
        }

        hall_match = re.search(r"HALLUCINATION_DETECTED:\s*(Yes|No)", text, re.IGNORECASE)
        if hall_match:
            result["hallucination_detected"] = hall_match.group(1).lower() == "yes"

        conf_match = re.search(r"CONFIDENCE:\s*(HIGH|MEDIUM|LOW)", text, re.IGNORECASE)
        if conf_match:
            result["confidence"] = conf_match.group(1).upper()

        flag_match = re.search(r"FLAG:\s*(.+)", text)
        if flag_match:
            result["flag"] = flag_match.group(1).strip()

        return result

    def format_verification_badge(self, metadata: dict) -> str:
        """Return a human-readable confidence badge for the UI."""
        confidence = metadata.get("confidence", "UNKNOWN")
        unsupported = metadata.get("unsupported_count", 0)

        badge_map = {
            "HIGH":    "✅ HIGH confidence — all claims source-grounded",
            "MEDIUM":  "🟡 MEDIUM confidence — mostly source-grounded",
            "LOW":     "🔴 LOW confidence — some claims could not be verified",
            "UNKNOWN": "⚪ Confidence unknown",
        }

        badge = badge_map.get(confidence, "⚪ Confidence unknown")
        if unsupported > 0:
            badge += f" ({unsupported} claim(s) softened)"
        return badge