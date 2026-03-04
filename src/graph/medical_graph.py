"""
medical_graph.py
----------------
LangGraph pipeline with structured JSON output, citation validation,
self-consistency, and verification.

Key design principle:
  citation-formatted text is stored in `citation_formatted_response` and
  carried through unchanged to the finalizer. The LLM rewrite stages
  (merger, self-consistency, verifier) operate on a plain-text summary
  for quality scoring only — they do NOT replace the cited output.
"""

import os
from typing import TypedDict, Optional, List, Annotated, Any
import operator

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv

from src.rag.vector_store import MedicalRetriever
from src.agents.diagnosis_agent import DiagnosisAgent
from src.agents.treatment_agent import TreatmentAgent
from src.agents.summarization_agent import SummarizationAgent
from src.agents.verifier_agent import VerifierAgent
from src.agents.self_consistency_agent import SelfConsistencyAgent
from src.utils.helpers import (
    detect_cancer_type_from_query, detect_intent,
    detect_comparison_query, detect_all_cancer_types,
    add_medical_disclaimer,
)
from src.utils.citation_validator import (
    validate_claims, filter_to_valid_claims,
    format_validated_response, MAX_REGENERATION_ATTEMPTS,
)
from src.utils.tracing import (
    trace_node, trace_agent, trace_graph_run,
    trace_citation_validation, setup_langsmith,
    create_run_metadata, build_tags, is_tracing_enabled,
)

load_dotenv()


# ─────────────────────────────────────────────
# State
# ─────────────────────────────────────────────

class MedicalAgentState(TypedDict):
    user_query:             str
    cancer_type:            Optional[str]
    intent:                 str
    is_comparison:          bool
    cancer_types_detected:  List[str]

    # Structured agent outputs
    diagnosis_structured:   Optional[Any]
    treatment_structured:   Optional[Any]
    summary_structured:     Optional[Any]

    # Citation-formatted text (never rewritten by downstream LLM calls)
    diagnosis_text:         Optional[str]
    treatment_text:         Optional[str]
    summary_text:           Optional[str]

    # The protected citation-formatted final response
    # This is built in merger and NEVER overwritten by verifier/self-consistency
    citation_formatted_response: Optional[str]

    # Validation
    diagnosis_valid:        bool
    treatment_valid:        bool
    summary_valid:          bool
    regeneration_count:     int
    validation_stats:       str

    # Quality pipeline (LLM rewrites for scoring only — not used as final output)
    merged_response:        Optional[str]
    consistency_score:      Optional[int]
    consistency_assessment: str
    verification_confidence: str
    unsupported_claims_count: int

    # Final assembled response
    final_response:         Optional[str]
    messages:               Annotated[List[str], operator.add]
    error:                  Optional[str]


# ─────────────────────────────────────────────
# Nodes
# ─────────────────────────────────────────────

def make_supervisor_node(retriever: MedicalRetriever):
    @trace_node("supervisor")
    def node(state: MedicalAgentState) -> MedicalAgentState:
        print("\n🎯 [Node: Supervisor]")
        query = state["user_query"]
        cancer_type = detect_cancer_type_from_query(query)
        intent      = detect_intent(query)
        is_comp     = detect_comparison_query(query)
        all_cancers = detect_all_cancer_types(query)
        print(f"   Cancer: {cancer_type or 'unknown'} | Intent: {intent} | Comparison: {is_comp}")
        return {
            **state,
            "cancer_type":                 cancer_type,
            "intent":                      intent,
            "is_comparison":               is_comp,
            "cancer_types_detected":       all_cancers,
            "diagnosis_structured":        None,
            "treatment_structured":        None,
            "summary_structured":          None,
            "diagnosis_text":              None,
            "treatment_text":              None,
            "summary_text":                None,
            "citation_formatted_response": None,
            "diagnosis_valid":             False,
            "treatment_valid":             False,
            "summary_valid":               False,
            "regeneration_count":          0,
            "validation_stats":            "",
            "merged_response":             None,
            "consistency_score":           None,
            "consistency_assessment":      "",
            "verification_confidence":     "UNKNOWN",
            "unsupported_claims_count":    0,
            "final_response":              None,
            "error":                       None,
        }
    return node


def make_diagnosis_node(retriever: MedicalRetriever):
    agent = DiagnosisAgent(retriever)

    @trace_node("diagnosis")
    def node(state: MedicalAgentState) -> MedicalAgentState:
        print("\n🔍 [Node: Diagnosis — JSON + Citation]")
        try:
            parsed, report, docs = agent.run(state["user_query"], state.get("cancer_type"))
            if parsed is None:
                return {**state, "diagnosis_text": "Diagnosis info unavailable.", "diagnosis_valid": False}

            valid_claims = filter_to_valid_claims(parsed.all_claims(), report)
            text  = format_validated_response(valid_claims, report, parsed.summary, parsed.cancer_type)
            stats = f"Diagnosis: {report.valid_claims}/{report.total_claims} claims cited"
            return {
                **state,
                "diagnosis_structured": parsed,
                "diagnosis_text":       text,
                "diagnosis_valid":      report.overall_valid,
                "validation_stats":     state.get("validation_stats", "") + stats + " | ",
            }
        except Exception as e:
            print(f"   ❌ {e}")
            return {**state, "diagnosis_text": f"Error: {e}", "diagnosis_valid": False, "error": str(e)}
    return node


def make_treatment_node(retriever: MedicalRetriever):
    agent = TreatmentAgent(retriever)

    @trace_node("treatment")
    def node(state: MedicalAgentState) -> MedicalAgentState:
        print("\n💊 [Node: Treatment — JSON + Citation]")
        try:
            parsed, report, docs = agent.run(state["user_query"], state.get("cancer_type"))
            if parsed is None:
                return {**state, "treatment_text": "Treatment info unavailable.", "treatment_valid": False}

            valid_claims = filter_to_valid_claims(parsed.all_claims(), report)
            text  = format_validated_response(valid_claims, report, parsed.summary, parsed.cancer_type)
            stats = f"Treatment: {report.valid_claims}/{report.total_claims} claims cited"
            return {
                **state,
                "treatment_structured": parsed,
                "treatment_text":       text,
                "treatment_valid":      report.overall_valid,
                "validation_stats":     state.get("validation_stats", "") + stats + " | ",
            }
        except Exception as e:
            print(f"   ❌ {e}")
            return {**state, "treatment_text": f"Error: {e}", "treatment_valid": False, "error": str(e)}
    return node


def make_summarization_node(retriever: MedicalRetriever):
    agent = SummarizationAgent(retriever)

    @trace_node("summarization")
    def node(state: MedicalAgentState) -> MedicalAgentState:
        print("\n📋 [Node: Summarization — JSON + Citation]")
        try:
            if state.get("is_comparison") and len(state.get("cancer_types_detected", [])) >= 2:
                parsed, report, docs = agent.run_comparison(
                    state["user_query"], state["cancer_types_detected"]
                )
            else:
                parsed, report, docs = agent.run(state["user_query"], state.get("cancer_type"))

            if parsed is None:
                return {**state, "summary_text": "Summary unavailable.", "summary_valid": False}

            valid_claims = filter_to_valid_claims(parsed.all_claims(), report)
            text  = format_validated_response(valid_claims, report, parsed.summary)
            stats = f"Summary: {report.valid_claims}/{report.total_claims} claims cited"
            return {
                **state,
                "summary_structured": parsed,
                "summary_text":       text,
                "summary_valid":      report.overall_valid,
                "validation_stats":   state.get("validation_stats", "") + stats + " | ",
            }
        except Exception as e:
            print(f"   ❌ {e}")
            return {**state, "summary_text": f"Error: {e}", "summary_valid": False, "error": str(e)}
    return node


def make_citation_validator_node(retriever: MedicalRetriever):
    @trace_node("citation_validator")
    def node(state: MedicalAgentState) -> MedicalAgentState:
        print("\n📎 [Node: Citation Validator]")
        all_valid = (
            state.get("diagnosis_valid", True) and
            state.get("treatment_valid", True) and
            state.get("summary_valid", True)
        )
        regen = state.get("regeneration_count", 0)

        if not all_valid and regen < MAX_REGENERATION_ATTEMPTS:
            print(f"   ⚠️  Triggering regeneration (attempt {regen + 1}/{MAX_REGENERATION_ATTEMPTS})")
            return {
                **state,
                "regeneration_count": regen + 1,
                "diagnosis_text":  None if not state.get("diagnosis_valid") else state.get("diagnosis_text"),
                "treatment_text":  None if not state.get("treatment_valid") else state.get("treatment_text"),
                "summary_text":    None if not state.get("summary_valid")  else state.get("summary_text"),
            }
        if not all_valid:
            print(f"   ⚠️  Max regenerations reached — proceeding with partial results")
        else:
            print(f"   ✅ Citations validated")
        return state
    return node


def make_merger_node(retriever: MedicalRetriever):
    """
    Builds BOTH:
    1. citation_formatted_response — the structured cited text (protected, never rewritten)
    2. merged_response — plain text summary for quality scoring by downstream agents
    """
    @trace_node("merger")
    def node(state: MedicalAgentState) -> MedicalAgentState:
        print("\n🔀 [Node: Merger]")

        diag  = state.get("diagnosis_text")
        treat = state.get("treatment_text")
        summ  = state.get("summary_text")

        # ── Build citation_formatted_response ─────────────────────
        # This is the final output — structured, cited, never touched again
        citation_sections = []

        if diag and treat:
            citation_sections.append("## Diagnosis\n")
            citation_sections.append(diag)
            citation_sections.append("\n## Treatment\n")
            citation_sections.append(treat)
        elif diag:
            citation_sections.append("## Diagnosis\n")
            citation_sections.append(diag)
        elif treat:
            citation_sections.append("## Treatment\n")
            citation_sections.append(treat)
        elif summ:
            citation_sections.append(summ)

        citation_formatted = "\n".join(citation_sections) if citation_sections else \
            "No verified information could be retrieved for your query."

        # ── Build merged_response (plain text for quality scoring) ─
        # Strip markdown so quality agents don't get confused by citation blocks
        import re
        plain_diag  = re.sub(r'>.*\n', '', diag or "")
        plain_treat = re.sub(r'>.*\n', '', treat or "")
        plain_summ  = re.sub(r'>.*\n', '', summ or "")
        plain_text  = "\n\n".join(filter(None, [plain_diag, plain_treat, plain_summ]))

        print(f"   ✅ Citation-formatted response built ({len(citation_formatted)} chars)")
        print(f"   ✅ Plain-text summary built for quality scoring ({len(plain_text)} chars)")

        return {
            **state,
            "citation_formatted_response": citation_formatted,
            "merged_response":             plain_text or citation_formatted,
        }
    return node


def make_self_consistency_node(retriever: MedicalRetriever):
    """Scores consistency on the plain-text summary. Does NOT replace citation output."""
    agent = SelfConsistencyAgent(retriever)

    @trace_node("self_consistency")
    def node(state: MedicalAgentState) -> MedicalAgentState:
        print("\n🔄 [Node: Self-Consistency — scoring only]")
        plain = state.get("merged_response", "")
        if not plain:
            return {**state, "consistency_score": None, "consistency_assessment": "skipped"}
        try:
            _, metadata = agent.run(
                query=state["user_query"],
                cancer_type=state.get("cancer_type"),
            )
            score      = metadata.get("consistency_score")
            assessment = metadata.get("assessment", "")
            print(f"   📊 Consistency score: {score}/100 — {assessment}")
            return {
                **state,
                "consistency_score":      score,
                "consistency_assessment": assessment,
            }
        except Exception as e:
            print(f"   ❌ {e} — skipping")
            return {**state, "consistency_score": None, "consistency_assessment": "error"}
    return node


def make_verifier_node(retriever: MedicalRetriever):
    """Scores source confidence on plain-text summary. Does NOT replace citation output."""
    agent = VerifierAgent(retriever)

    @trace_node("verifier")
    def node(state: MedicalAgentState) -> MedicalAgentState:
        print("\n✅ [Node: Verifier — scoring only]")
        plain = state.get("merged_response", "")
        if not plain:
            return {**state, "verification_confidence": "UNKNOWN", "unsupported_claims_count": 0}
        try:
            _, metadata = agent.full_verify(
                response=plain,
                query=state["user_query"],
                cancer_type=state.get("cancer_type"),
            )
            conf        = metadata.get("confidence", "UNKNOWN")
            unsupported = metadata.get("unsupported_count", 0)
            print(f"   🔒 Confidence: {conf} | Unsupported in plain text: {unsupported}")
            return {
                **state,
                "verification_confidence":  conf,
                "unsupported_claims_count": unsupported,
            }
        except Exception as e:
            print(f"   ❌ {e} — skipping")
            return {**state, "verification_confidence": "UNKNOWN", "unsupported_claims_count": 0}
    return node


def make_finalizer_node():
    """
    Assembles final response:
    - Uses citation_formatted_response as the body (never rewritten)
    - Appends quality scores from self-consistency and verifier
    - Appends medical disclaimer
    """
    @trace_node("finalizer")
    def node(state: MedicalAgentState) -> MedicalAgentState:
        print("\n📤 [Node: Finalizer]")

        # Always use the citation-formatted body
        body = state.get("citation_formatted_response") or \
               "No verified information could be retrieved for your query."

        # Quality footer
        footer = []

        val_stats = state.get("validation_stats", "").rstrip(" | ")
        if val_stats:
            footer.append(f"📎 **Citations:** {val_stats}")

        score = state.get("consistency_score")
        if score is not None:
            icon = "✅" if score >= 80 else ("🟡" if score >= 55 else "🔴")
            footer.append(f"{icon} **Self-Consistency:** {score}/100")

        conf = state.get("verification_confidence", "UNKNOWN")
        if conf != "UNKNOWN":
            icon = {"HIGH": "✅", "MEDIUM": "🟡", "LOW": "🔴"}.get(conf, "⚪")
            footer.append(f"{icon} **Source Verification:** {conf}")

        unsupported = state.get("unsupported_claims_count", 0)
        if unsupported > 0:
            footer.append(f"✂️ **Flagged:** {unsupported} unsupported claim(s) in summary")

        regen = state.get("regeneration_count", 0)
        if regen > 0:
            footer.append(f"🔄 **Regenerations:** {regen}")

        final = add_medical_disclaimer(body)
        if footer:
            final += "\n\n---\n**🛡️ Response Quality**\n" + "\n".join(footer)

        return {**state, "final_response": final}
    return node


# ─────────────────────────────────────────────
# Routing
# ─────────────────────────────────────────────

def route_after_supervisor(state: MedicalAgentState) -> str:
    if state.get("is_comparison") and len(state.get("cancer_types_detected", [])) >= 2:
        return "summarization"
    intent = state.get("intent", "general")
    if intent == "diagnosis":  return "diagnosis"
    if intent == "treatment":  return "treatment"
    return "summarization"


def route_after_diagnosis(state: MedicalAgentState) -> str:
    return "treatment" if state.get("intent") in ("treatment", "multi") else "citation_validator"


def route_after_treatment(state: MedicalAgentState) -> str:
    return "citation_validator"


def route_after_summarization(state: MedicalAgentState) -> str:
    return "citation_validator"


def route_after_citation_validator(state: MedicalAgentState) -> str:
    regen = state.get("regeneration_count", 0)
    if regen > 0 and regen <= MAX_REGENERATION_ATTEMPTS:
        if not state.get("diagnosis_valid") and state.get("diagnosis_text") is None:
            return "diagnosis"
        if not state.get("treatment_valid") and state.get("treatment_text") is None:
            return "treatment"
        if not state.get("summary_valid") and state.get("summary_text") is None:
            return "summarization"
    return "merger"


# ─────────────────────────────────────────────
# Graph Builder
# ─────────────────────────────────────────────

def build_medical_graph(retriever: MedicalRetriever):
    print("\n🔨 Building LangGraph (Citation-Protected Pipeline)...")

    graph = StateGraph(MedicalAgentState)

    graph.add_node("supervisor",         make_supervisor_node(retriever))
    graph.add_node("diagnosis",          make_diagnosis_node(retriever))
    graph.add_node("treatment",          make_treatment_node(retriever))
    graph.add_node("summarization",      make_summarization_node(retriever))
    graph.add_node("citation_validator", make_citation_validator_node(retriever))
    graph.add_node("merger",             make_merger_node(retriever))
    graph.add_node("self_consistency",   make_self_consistency_node(retriever))
    graph.add_node("verifier",           make_verifier_node(retriever))
    graph.add_node("finalizer",          make_finalizer_node())

    graph.set_entry_point("supervisor")

    graph.add_conditional_edges("supervisor", route_after_supervisor, {
        "diagnosis": "diagnosis", "treatment": "treatment", "summarization": "summarization",
    })
    graph.add_conditional_edges("diagnosis", route_after_diagnosis, {
        "treatment": "treatment", "citation_validator": "citation_validator",
    })
    graph.add_conditional_edges("treatment",     route_after_treatment,     {"citation_validator": "citation_validator"})
    graph.add_conditional_edges("summarization", route_after_summarization, {"citation_validator": "citation_validator"})
    graph.add_conditional_edges("citation_validator", route_after_citation_validator, {
        "diagnosis":    "diagnosis",
        "treatment":    "treatment",
        "summarization":"summarization",
        "merger":       "merger",
    })

    graph.add_edge("merger",          "self_consistency")
    graph.add_edge("self_consistency", "verifier")
    graph.add_edge("verifier",         "finalizer")
    graph.add_edge("finalizer",        END)

    compiled = graph.compile()
    print("✅ LangGraph compiled — citation output is protected from LLM rewrites")
    return compiled


@trace_graph_run
def run_graph(
    graph,
    query: str,
    conversation_history: Optional[List[str]] = None,
) -> str:
    if graph is None:
        raise ValueError("Graph not initialized. Call build_medical_graph() first.")

    initial: MedicalAgentState = {
        "user_query":                  query,
        "cancer_type":                 None,
        "intent":                      "general",
        "is_comparison":               False,
        "cancer_types_detected":       [],
        "diagnosis_structured":        None,
        "treatment_structured":        None,
        "summary_structured":          None,
        "diagnosis_text":              None,
        "treatment_text":              None,
        "summary_text":                None,
        "citation_formatted_response": None,
        "diagnosis_valid":             False,
        "treatment_valid":             False,
        "summary_valid":               False,
        "regeneration_count":          0,
        "validation_stats":            "",
        "merged_response":             None,
        "consistency_score":           None,
        "consistency_assessment":      "",
        "verification_confidence":     "UNKNOWN",
        "unsupported_claims_count":    0,
        "final_response":              None,
        "messages":                    conversation_history or [],
        "error":                       None,
    }

    print(f"\n{'='*55}")
    print(f"🏥 LANGGRAPH MEDICAL RESEARCH ASSISTANT")
    print(f"{'='*55}")
    print(f"Query: {query}\n")

    final = graph.invoke(initial)
    return final.get("final_response", "An error occurred during processing.")