"""
self_consistency_agent.py
--------------------------
Self-Consistency Agent: Generates two independent answers to the same
query using different prompting strategies, then compares them and
extracts only the overlapping/agreed statements.

Anti-hallucination layer #2.

Based on: Wang et al. (2022) "Self-Consistency Improves Chain of Thought
Reasoning in Language Models"
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

# ── Answer Generation Prompts (two distinct reasoning styles) ──────────────

ANSWER_A_SYSTEM = """You are a careful medical information specialist.
Answer the question by reasoning step-by-step through the provided context.
Focus on what is EXPLICITLY stated in the NCCN guidelines.
Be conservative — only state what you can directly support from the text.
"""

ANSWER_B_SYSTEM = """You are a clinical medical educator summarizing NCCN guidelines.
Answer the question by identifying the key facts in the context and 
organizing them clearly for a patient audience.
Focus on practical, actionable information from the source material.
"""

ANSWER_PROMPT = """Using ONLY the medical context below, answer this question.
For every factual claim, include the source file name, page number, and a short quote.

**Question:** {query}

**NCCN Guidelines Context:**
{context}

For each claim, format it as:
**[claim statement]**

> 📄 **Source:** `[source_file from context metadata]`
> 📖 **Page:** [page_number from context metadata]
> 🗂️ **Chapter:** [chapter from context metadata]
> 💬 **Quote:** "[short verbatim quote from context]"

Base every claim strictly on the provided context. Include page numbers and file names.
"""

# ── Comparison / Overlap Extraction Prompt ─────────────────────────────────

DEBATE_SYSTEM = """You are a Medical Consensus Arbiter.
You will receive two independently generated answers to the same medical question.
Your job is to:

1. Identify statements that BOTH answers agree on (consensus statements)
2. Identify statements that only ONE answer makes (divergent claims)
3. Identify any direct contradictions between the answers

Then produce a final consensus response that:
- Includes ONLY statements agreed upon by both answers
- Marks uncertain or single-source claims clearly with "(unconfirmed)"
- Is clearly structured and medically accurate
- Does NOT introduce any new information not in either answer

CRITICAL FORMATTING RULE:
Both answers contain structured citation blocks in this format:
  > 📄 **Source:** `filename.pdf`
  > 📖 **Page:** N
  > 🗂️ **Chapter:** ...
  > 💬 **Quote:** "..."

You MUST preserve ALL citation blocks exactly as-is in the consensus response.
Do NOT replace them with "(Source 1)" or any other shorthand.
Do NOT rewrite or strip citation metadata.
If a claim from Answer A and Answer B agree, keep the citation block from Answer A.
"""

DEBATE_PROMPT = """Compare these two answers. Extract consensus. PRESERVE all citation blocks.

**Original Question:** {query}

**Answer A (Step-by-step reasoning):**
{answer_a}

**Answer B (Key facts summary):**
{answer_b}

Produce:
---CONSENSUS ANALYSIS---
Agreed Statements: [bullet list of what both answers agree on]
Divergent Claims: [what only one answer says — mark as unconfirmed]
Contradictions: [any direct conflicts — flag these]

---CONSENSUS RESPONSE---
[Final response using only agreed statements.
Copy citation blocks (📄/📖/🗂️/💬 lines) exactly from the source answers.
Do NOT rewrite or summarize citation metadata.]

---CONSISTENCY SCORE---
Score: [0-100] (how much the two answers agreed)
Assessment: [one sentence]
"""


class SelfConsistencyAgent:
    """
    Generates two independent answers using different reasoning strategies,
    then uses a debate/arbiter step to extract only the overlapping claims.
    
    This significantly reduces hallucination by requiring two independent
    reasoning paths to agree before a statement is included in the output.
    """

    def __init__(self, retriever: MedicalRetriever):
        self.retriever = retriever

        # Answer generator A — step-by-step reasoning style
        self.llm_a = ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            temperature=0.1,   # Slight variation to encourage different reasoning
            max_tokens=1500,
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )

        # Answer generator B — key facts / educator style
        self.llm_b = ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            temperature=0.2,   # More variation for B
            max_tokens=1500,
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )

        # Arbiter — always deterministic
        self.llm_arbiter = ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            temperature=0.0,
            max_tokens=2000,
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )

        self.name = "Self-Consistency Agent"

    def run(
        self,
        query: str,
        cancer_type: Optional[str] = None,
        source_docs: Optional[List[Document]] = None,
        k: int = 6,
    ) -> Tuple[str, dict]:
        """
        Full self-consistency pipeline:
        1. Retrieve context
        2. Generate Answer A (step-by-step)
        3. Generate Answer B (key facts)
        4. Arbiter compares and extracts consensus

        Returns:
            (consensus_response, consistency_metadata)
        """
        log_agent_call(self.name, query, cancer_type)

        # Step 1: Get source documents
        if not source_docs:
            source_docs = self.retriever.retrieve(
                query=query,
                cancer_type=cancer_type,
                k=k,
            )
        context = self.retriever.format_context(source_docs)

        # Step 2: Generate Answer A
        print(f"   🅰️  Generating Answer A (step-by-step reasoning)...")
        answer_a = self._generate_answer(
            query=query,
            context=context,
            system_prompt=ANSWER_A_SYSTEM,
            llm=self.llm_a,
        )

        # Step 3: Generate Answer B
        print(f"   🅱️  Generating Answer B (key facts summary)...")
        answer_b = self._generate_answer(
            query=query,
            context=context,
            system_prompt=ANSWER_B_SYSTEM,
            llm=self.llm_b,
        )

        # Step 4: Arbiter extracts consensus
        print(f"   ⚖️  Arbiter extracting consensus...")
        consensus_response, metadata = self._run_arbiter(
            query=query,
            answer_a=answer_a,
            answer_b=answer_b,
        )

        print(f"   ✅ Self-Consistency done | Score: {metadata.get('consistency_score', '?')}/100")

        return consensus_response, {
            **metadata,
            "answer_a": answer_a,
            "answer_b": answer_b,
        }

    def _generate_answer(
        self,
        query: str,
        context: str,
        system_prompt: str,
        llm: ChatGroq,
    ) -> str:
        """Generate a single independent answer."""
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=ANSWER_PROMPT.format(
                query=query,
                context=context,
            )),
        ]
        response = llm.invoke(messages)
        return clean_llm_response(response.content)

    def _run_arbiter(
        self,
        query: str,
        answer_a: str,
        answer_b: str,
    ) -> Tuple[str, dict]:
        """Run the debate/arbiter step to extract consensus."""
        messages = [
            SystemMessage(content=DEBATE_SYSTEM),
            HumanMessage(content=DEBATE_PROMPT.format(
                query=query,
                answer_a=answer_a,
                answer_b=answer_b,
            )),
        ]

        response = self.llm_arbiter.invoke(messages)
        raw_text = clean_llm_response(response.content)

        metadata = self._parse_debate_output(raw_text)
        consensus = self._extract_consensus_response(raw_text, answer_a)

        return consensus, metadata

    def _parse_debate_output(self, text: str) -> dict:
        """Parse consistency score and assessment from arbiter output."""
        metadata = {
            "consistency_score": None,
            "assessment": "",
            "has_contradictions": False,
            "divergent_claims": [],
        }

        # Extract consistency score
        score_match = re.search(r"Score:\s*(\d+)", text)
        if score_match:
            metadata["consistency_score"] = int(score_match.group(1))

        # Extract assessment
        assess_match = re.search(r"Assessment:\s*(.+)", text)
        if assess_match:
            metadata["assessment"] = assess_match.group(1).strip()

        # Check for contradictions
        if "contradiction" in text.lower() and "none" not in text.lower():
            metadata["has_contradictions"] = True

        return metadata

    def _extract_consensus_response(self, full_text: str, fallback: str) -> str:
        """Extract the consensus response section from arbiter output."""
        if "---CONSENSUS RESPONSE---" in full_text:
            parts = full_text.split("---CONSENSUS RESPONSE---")
            if len(parts) > 1:
                # Strip the consistency score section if present
                consensus_part = parts[1]
                if "---CONSISTENCY SCORE---" in consensus_part:
                    consensus_part = consensus_part.split("---CONSISTENCY SCORE---")[0]
                return clean_llm_response(consensus_part)
        return fallback

    def format_consistency_badge(self, metadata: dict) -> str:
        """Return a human-readable consistency badge for the UI."""
        score = metadata.get("consistency_score")
        has_contradictions = metadata.get("has_contradictions", False)

        if score is None:
            return "⚪ Consistency score unavailable"

        if score >= 80:
            badge = f"✅ HIGH consistency ({score}/100) — both reasoning paths agreed"
        elif score >= 55:
            badge = f"🟡 MEDIUM consistency ({score}/100) — partial agreement"
        else:
            badge = f"🔴 LOW consistency ({score}/100) — answers diverged significantly"

        if has_contradictions:
            badge += " ⚠️ contradictions detected & resolved"

        return badge