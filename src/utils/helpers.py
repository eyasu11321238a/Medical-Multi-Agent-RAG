"""
helpers.py
----------
Shared utilities for the multi-agent medical research assistant.
"""

import re
import os
from typing import Optional, List, Dict, Any


# ─────────────────────────────────────────────
# Cancer Type Detection
# ─────────────────────────────────────────────

CANCER_KEYWORDS: Dict[str, List[str]] = {
    "Basal Cell Skin Cancer": [
        "basal cell", "bcc", "basal cell carcinoma", "basal cell cancer"
    ],
    "Melanoma": [
        "melanoma", "malignant melanoma", "skin melanoma"
    ],
    "Squamous Cell Skin Cancer": [
        "squamous cell", "scc", "squamous cell carcinoma", "squamous cell cancer"
    ],
    "Breast Cancer": [
        "breast cancer", "breast carcinoma", "breast tumor", "mammary"
    ],
    "Lung Cancer": [
        "lung cancer", "lung carcinoma", "nsclc", "sclc",
        "non-small cell", "small cell lung"
    ],
    "Colorectal Cancer": [
        "colorectal", "colon cancer", "rectal cancer",
        "colorectal carcinoma", "bowel cancer"
    ],
    "Prostate Cancer": [
        "prostate cancer", "prostate carcinoma", "prostate tumor"
    ],
    "Leukemia": [
        "leukemia", "leukaemia", "blood cancer", "aml", "cml", "all", "cll"
    ],
}

INTENT_KEYWORDS: Dict[str, List[str]] = {
    "diagnosis": [
        "what is", "what are signs", "symptoms", "signs of", "do i have",
        "how do i know", "identify", "diagnose", "diagnosis", "detect",
        "risk factors", "causes", "who gets"
    ],
    "treatment": [
        "treatment", "treat", "surgery", "therapy", "drug", "medication",
        "mohs", "chemotherapy", "radiation", "immunotherapy", "targeted",
        "clinical trial", "options for", "how to treat", "cure"
    ],
    "summarization": [
        "summarize", "summary", "explain", "what is", "describe",
        "compare", "difference between", "vs", "versus", "overview",
        "tell me about", "how does", "what does"
    ],
}


def detect_cancer_type_from_query(query: str) -> Optional[str]:
    """
    Detect cancer type mentioned in user query.
    Returns the matched cancer type label or None.
    """
    query_lower = query.lower()
    for cancer_type, keywords in CANCER_KEYWORDS.items():
        if any(kw in query_lower for kw in keywords):
            return cancer_type
    return None


def detect_intent(query: str) -> str:
    """
    Detect the primary intent of the user query.
    Returns: 'diagnosis', 'treatment', 'summarization', or 'general'
    """
    query_lower = query.lower()
    scores: Dict[str, int] = {intent: 0 for intent in INTENT_KEYWORDS}

    for intent, keywords in INTENT_KEYWORDS.items():
        for kw in keywords:
            if kw in query_lower:
                scores[intent] += 1

    # Return highest scoring intent
    best_intent = max(scores, key=lambda k: scores[k])
    return best_intent if scores[best_intent] > 0 else "general"


def detect_comparison_query(query: str) -> bool:
    """Check if the query is asking for a comparison between cancer types."""
    comparison_keywords = ["compare", "vs", "versus", "difference between",
                           "contrast", "similarities", "both"]
    query_lower = query.lower()
    return any(kw in query_lower for kw in comparison_keywords)


def detect_all_cancer_types(query: str) -> List[str]:
    """Detect ALL cancer types mentioned in a query (for comparison queries)."""
    query_lower = query.lower()
    detected = []
    for cancer_type, keywords in CANCER_KEYWORDS.items():
        if any(kw in query_lower for kw in keywords):
            detected.append(cancer_type)
    return detected


# ─────────────────────────────────────────────
# Response Formatting
# ─────────────────────────────────────────────

def format_final_response(
    intent: str,
    cancer_type: Optional[str],
    diagnosis_output: Optional[str] = None,
    treatment_output: Optional[str] = None,
    summary_output: Optional[str] = None,
) -> str:
    """
    Merge outputs from multiple agents into a final structured response.
    """
    sections = []

    if cancer_type:
        sections.append(f"**Cancer Type Identified:** {cancer_type}\n")

    if diagnosis_output:
        sections.append(f"### 🔍 Diagnosis Information\n{diagnosis_output}")

    if treatment_output:
        sections.append(f"### 💊 Treatment Options\n{treatment_output}")

    if summary_output:
        sections.append(f"### 📋 Summary\n{summary_output}")

    if not sections:
        return "I could not find relevant information for your query. Please try rephrasing."

    separator = "\n\n" + "─" * 50 + "\n\n"
    return separator.join(sections)


def clean_llm_response(text: str) -> str:
    """Clean up common LLM response artifacts."""
    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def add_medical_disclaimer(response: str) -> str:
    """Append standard medical disclaimer to response."""
    disclaimer = (
        "\n\n---\n"
        "⚠️ **Medical Disclaimer:** This information is based on NCCN patient "
        "guidelines and is for educational purposes only. It is not a substitute "
        "for professional medical advice, diagnosis, or treatment. Always consult "
        "a qualified healthcare provider for personal medical guidance."
    )
    return response + disclaimer


# ─────────────────────────────────────────────
# Logging Utilities
# ─────────────────────────────────────────────

def log_agent_call(agent_name: str, query: str, cancer_type: Optional[str]) -> None:
    """Log agent activation for debugging."""
    print(f"\n{'─'*50}")
    print(f"🤖 Agent Activated : {agent_name}")
    print(f"   Query          : {query[:80]}{'...' if len(query) > 80 else ''}")
    print(f"   Cancer Type    : {cancer_type or 'Not specified'}")
    print(f"{'─'*50}")


def log_graph_state(state: Dict[str, Any]) -> None:
    """Log current LangGraph state."""
    print("\n📊 Current Graph State:")
    for key, value in state.items():
        if isinstance(value, str) and len(value) > 60:
            print(f"   {key}: {value[:60]}...")
        elif isinstance(value, list):
            print(f"   {key}: [{len(value)} items]")
        else:
            print(f"   {key}: {value}")
