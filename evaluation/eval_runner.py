"""
eval_runner.py
--------------
Runs the 20-question evaluation dataset through two pipeline variants:
  A) WITH verifier agent   (full pipeline)
  B) WITHOUT verifier agent (verifier node bypassed)

For each question, records:
  - citation_accuracy    : valid_claims / total_claims (from citation validator)
  - unsupported_rate     : unsupported_claims_count / total_claims_in_response
  - regeneration_count   : how many times citation validator triggered regen
  - verification_conf    : HIGH / MEDIUM / LOW / UNKNOWN (verifier output)
  - concept_hit_rate     : % of expected_concepts found in final response
  - answer_length        : chars in final response (proxy for completeness)
  - error                : any exception raised

Outputs a JSON results file and prints a summary comparison table.
"""

import os
import sys
import json
import time
import re
from copy import deepcopy
from typing import Dict, Any, Optional
from dataclasses import asdict

# ── path setup ────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from evaluation.eval_dataset import EVAL_DATASET, EvalQuestion

# ─────────────────────────────────────────────────────────────────
# Results data structure
# ─────────────────────────────────────────────────────────────────

def empty_result(qid: int, question: str, variant: str) -> Dict:
    return {
        "question_id":        qid,
        "question":           question[:80],
        "variant":            variant,      # "with_verifier" | "without_verifier"
        "citation_accuracy":  0.0,          # 0.0 – 1.0
        "total_claims":       0,
        "valid_claims":       0,
        "unsupported_count":  0,
        "unsupported_rate":   0.0,
        "regeneration_count": 0,
        "verification_conf":  "N/A",
        "concept_hit_rate":   0.0,
        "concepts_found":     [],
        "concepts_missing":   [],
        "answer_length":      0,
        "latency_s":          0.0,
        "error":              None,
    }


# ─────────────────────────────────────────────────────────────────
# Pipeline initialisation (cached)
# ─────────────────────────────────────────────────────────────────

_graph_with    = None
_graph_without = None
_retriever     = None


def _init_pipeline():
    global _graph_with, _graph_without, _retriever

    if _graph_with and _graph_without:
        return

    print("\n🔧 Initialising pipeline...")
    from src.rag.vector_store import MedicalRetriever, get_or_build_vector_store
    from src.graph.medical_graph import build_medical_graph

    faiss_path = os.path.join(PROJECT_ROOT, "data", "faiss_index")
    if not os.path.exists(os.path.join(faiss_path, "index.faiss")):
        raise RuntimeError(
            f"FAISS index not found at {faiss_path}.\n"
            "Run ingestion first:\n"
            "  python main.py --ingest"
        )

    print("   Loading FAISS index...")
    vector_store = get_or_build_vector_store()   # loads from FAISS_PATH in .env
    _retriever   = MedicalRetriever(vector_store)

    # WITH verifier — standard graph
    _graph_with = build_medical_graph(_retriever)

    # WITHOUT verifier — same graph but verifier node is patched to be a passthrough
    _graph_without = _build_graph_no_verifier(_retriever)

    print("✅ Both pipeline variants ready\n")


def _build_graph_no_verifier(retriever):
    """Build the graph with the verifier node replaced by a passthrough."""
    from langgraph.graph import StateGraph, END
    from src.graph.medical_graph import (
        MedicalAgentState,
        make_supervisor_node, make_diagnosis_node, make_treatment_node,
        make_summarization_node, make_citation_validator_node,
        make_merger_node, make_self_consistency_node, make_finalizer_node,
        route_after_supervisor, route_after_diagnosis, route_after_treatment,
        route_after_summarization, route_after_citation_validator,
    )
    from src.utils.citation_validator import MAX_REGENERATION_ATTEMPTS

    def passthrough_verifier(state: MedicalAgentState) -> MedicalAgentState:
        """Bypass verifier: keep response untouched, mark confidence SKIPPED."""
        return {
            **state,
            "verification_confidence":  "SKIPPED",
            "unsupported_claims_count": 0,
        }

    graph = StateGraph(MedicalAgentState)
    graph.add_node("supervisor",         make_supervisor_node(retriever))
    graph.add_node("diagnosis",          make_diagnosis_node(retriever))
    graph.add_node("treatment",          make_treatment_node(retriever))
    graph.add_node("summarization",      make_summarization_node(retriever))
    graph.add_node("citation_validator", make_citation_validator_node(retriever))
    graph.add_node("merger",             make_merger_node(retriever))
    graph.add_node("self_consistency",   make_self_consistency_node(retriever))
    graph.add_node("verifier",           passthrough_verifier)       # ← patched
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
        "diagnosis": "diagnosis", "treatment": "treatment",
        "summarization": "summarization", "merger": "merger",
    })
    graph.add_edge("merger",           "self_consistency")
    graph.add_edge("self_consistency",  "verifier")
    graph.add_edge("verifier",          "finalizer")
    graph.add_edge("finalizer",         END)

    return graph.compile()


# ─────────────────────────────────────────────────────────────────
# Single question runner
# ─────────────────────────────────────────────────────────────────

def _run_one(graph, q: EvalQuestion, variant: str) -> Dict:
    result = empty_result(q.id, q.question, variant)
    t0 = time.time()

    try:
        from src.graph.medical_graph import MedicalAgentState

        initial: MedicalAgentState = {
            "user_query":                  q.question,
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
            "messages":                    [],
            "error":                       None,
        }

        final_state = graph.invoke(initial)

        # ── Extract raw metrics from state ─────────────────────────
        val_stats  = final_state.get("validation_stats", "")
        final_text = final_state.get("final_response", "") or ""
        regen      = final_state.get("regeneration_count", 0)
        conf       = final_state.get("verification_confidence", "UNKNOWN")
        unsup      = final_state.get("unsupported_claims_count", 0)

        # Parse citation accuracy from validation_stats string
        # Format: "Treatment: 3/5 claims cited | ..."
        total_claims = 0
        valid_claims  = 0
        for match in re.finditer(r"(\d+)/(\d+) claims", val_stats):
            valid_claims  += int(match.group(1))
            total_claims  += int(match.group(2))

        citation_acc = round(valid_claims / total_claims, 3) if total_claims > 0 else 0.0

        # unsupported_rate relative to total claims
        unsup_rate = round(unsup / max(total_claims, 1), 3)

        # Concept hit rate — how many expected concepts appear in final answer
        answer_lower = final_text.lower()
        found   = [c for c in q.expected_concepts if c.lower() in answer_lower]
        missing = [c for c in q.expected_concepts if c.lower() not in answer_lower]
        concept_rate = round(len(found) / len(q.expected_concepts), 3) if q.expected_concepts else 1.0

        result.update({
            "citation_accuracy":  citation_acc,
            "total_claims":       total_claims,
            "valid_claims":       valid_claims,
            "unsupported_count":  unsup,
            "unsupported_rate":   unsup_rate,
            "regeneration_count": regen,
            "verification_conf":  conf,
            "concept_hit_rate":   concept_rate,
            "concepts_found":     found,
            "concepts_missing":   missing,
            "answer_length":      len(final_text),
            "latency_s":          round(time.time() - t0, 2),
        })

    except Exception as e:
        result["error"]      = str(e)
        result["latency_s"]  = round(time.time() - t0, 2)
        print(f"   ❌ Q{q.id} error: {e}")

    return result


# ─────────────────────────────────────────────────────────────────
# Full evaluation run
# ─────────────────────────────────────────────────────────────────

def run_evaluation(
    questions=None,
    output_path: str = None,
    verbose: bool = True,
) -> Dict:
    """
    Run all 20 questions through both variants.
    Returns dict with keys "with_verifier", "without_verifier", "comparison".
    Saves JSON to output_path if provided.
    """
    _init_pipeline()

    qs = questions or EVAL_DATASET
    results_with    = []
    results_without = []

    total = len(qs)
    for i, q in enumerate(qs, 1):
        print(f"\n{'─'*55}")
        print(f"Q{q.id:02d}/{total} [{q.category}] {q.question[:60]}...")

        # WITH verifier
        print(f"  ▶ Running WITH verifier...")
        r_with = _run_one(_graph_with, q, "with_verifier")
        results_with.append(r_with)
        if verbose:
            _print_row(r_with)

        # WITHOUT verifier
        print(f"  ▶ Running WITHOUT verifier...")
        r_without = _run_one(_graph_without, q, "without_verifier")
        results_without.append(r_without)
        if verbose:
            _print_row(r_without)

    comparison = _build_comparison(results_with, results_without)

    output = {
        "with_verifier":    results_with,
        "without_verifier": results_without,
        "comparison":       comparison,
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n💾 Results saved to {output_path}")

    _print_comparison_table(comparison)
    return output


# ─────────────────────────────────────────────────────────────────
# Comparison computation
# ─────────────────────────────────────────────────────────────────

def _avg(lst, key):
    vals = [r[key] for r in lst if r.get("error") is None]
    return round(sum(vals) / len(vals), 3) if vals else 0.0


def _build_comparison(with_v, without_v) -> Dict:
    """Compute aggregate metrics for both variants."""
    def agg(results):
        n       = len(results)
        errors  = sum(1 for r in results if r.get("error"))
        regen_q = sum(1 for r in results if r["regeneration_count"] > 0)
        conf_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "UNKNOWN": 0, "SKIPPED": 0}
        for r in results:
            c = r.get("verification_conf", "UNKNOWN")
            conf_counts[c] = conf_counts.get(c, 0) + 1

        return {
            "n_questions":            n,
            "errors":                 errors,
            "avg_citation_accuracy":  _avg(results, "citation_accuracy"),
            "avg_concept_hit_rate":   _avg(results, "concept_hit_rate"),
            "avg_unsupported_rate":   _avg(results, "unsupported_rate"),
            "avg_regeneration_count": _avg(results, "regeneration_count"),
            "questions_with_regen":   regen_q,
            "regen_rate":             round(regen_q / n, 3),
            "avg_latency_s":          _avg(results, "latency_s"),
            "avg_answer_length":      _avg(results, "answer_length"),
            "avg_total_claims":       _avg(results, "total_claims"),
            "avg_valid_claims":       _avg(results, "valid_claims"),
            "verification_conf_dist": conf_counts,
        }

    with_agg    = agg(with_v)
    without_agg = agg(without_v)

    def delta(key):
        a = with_agg[key]
        b = without_agg[key]
        if isinstance(a, float):
            return round(a - b, 3)
        return a - b

    return {
        "with_verifier":    with_agg,
        "without_verifier": without_agg,
        "delta": {
            "citation_accuracy_delta":  delta("avg_citation_accuracy"),
            "concept_hit_rate_delta":   delta("avg_concept_hit_rate"),
            "unsupported_rate_delta":   delta("avg_unsupported_rate"),
            "regeneration_rate_delta":  delta("regen_rate"),
            "latency_delta_s":          delta("avg_latency_s"),
        },
    }


# ─────────────────────────────────────────────────────────────────
# Console output
# ─────────────────────────────────────────────────────────────────

def _print_row(r: Dict):
    status = "❌ ERROR" if r.get("error") else "✅"
    print(
        f"    {status} cit={r['citation_accuracy']:.0%} "
        f"concept={r['concept_hit_rate']:.0%} "
        f"unsup={r['unsupported_rate']:.0%} "
        f"regen={r['regeneration_count']} "
        f"conf={r['verification_conf']} "
        f"{r['latency_s']:.1f}s"
    )


def _print_comparison_table(comparison: Dict):
    W = comparison["with_verifier"]
    N = comparison["without_verifier"]
    D = comparison["delta"]

    print("\n")
    print("=" * 72)
    print("  EVALUATION RESULTS — WITH vs WITHOUT VERIFIER AGENT")
    print("=" * 72)
    print(f"  {'Metric':<32} {'Without':>10} {'With':>10} {'Delta':>10}")
    print("─" * 72)

    rows = [
        ("Citation Accuracy",      f"{N['avg_citation_accuracy']:.1%}", f"{W['avg_citation_accuracy']:.1%}", f"{D['citation_accuracy_delta']:+.1%}"),
        ("Concept Hit Rate",        f"{N['avg_concept_hit_rate']:.1%}",   f"{W['avg_concept_hit_rate']:.1%}",   f"{D['concept_hit_rate_delta']:+.1%}"),
        ("Unsupported Claim Rate",  f"{N['avg_unsupported_rate']:.1%}",   f"{W['avg_unsupported_rate']:.1%}",   f"{D['unsupported_rate_delta']:+.1%}"),
        ("Regeneration Rate",       f"{N['regen_rate']:.1%}",             f"{W['regen_rate']:.1%}",             f"{D['regeneration_rate_delta']:+.1%}"),
        ("Avg Claims / Question",   f"{N['avg_total_claims']:.1f}",       f"{W['avg_total_claims']:.1f}",       ""),
        ("Avg Valid Claims",        f"{N['avg_valid_claims']:.1f}",       f"{W['avg_valid_claims']:.1f}",       ""),
        ("Avg Latency (s)",         f"{N['avg_latency_s']:.1f}",          f"{W['avg_latency_s']:.1f}",          f"{D['latency_delta_s']:+.1f}"),
        ("Avg Answer Length (ch)",  f"{N['avg_answer_length']:.0f}",      f"{W['avg_answer_length']:.0f}",      ""),
        ("Errors",                  f"{N['errors']}",                     f"{W['errors']}",                     ""),
    ]

    for label, n_val, w_val, d_val in rows:
        print(f"  {label:<32} {n_val:>10} {w_val:>10} {d_val:>10}")

    print("─" * 72)

    # Verifier confidence distribution
    wc = W["verification_conf_dist"]
    nc = N["verification_conf_dist"]
    print(f"\n  Verification Confidence Distribution:")
    print(f"  {'Level':<12} {'Without':>12} {'With':>12}")
    print("  " + "─" * 38)
    for level in ["HIGH", "MEDIUM", "LOW", "UNKNOWN", "SKIPPED"]:
        print(f"  {level:<12} {nc.get(level,0):>12} {wc.get(level,0):>12}")

    print("=" * 72)
    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run NCCN evaluation")
    parser.add_argument("--output", default="evaluation/results/eval_results.json",
                        help="Path to save JSON results")
    parser.add_argument("--quick", action="store_true",
                        help="Run first 5 questions only (smoke test)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    qs = EVAL_DATASET[:5] if args.quick else EVAL_DATASET
    run_evaluation(questions=qs, output_path=args.output)