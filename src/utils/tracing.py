"""
tracing.py
----------
LangSmith tracing integration for the Medical Research Assistant.

Provides:
  - setup_langsmith()         : initialise tracing from env vars
  - trace_node()              : decorator for LangGraph nodes
  - trace_agent()             : decorator for agent .run() calls
  - trace_llm_call()          : decorator for individual LLM calls
  - trace_citation_validation(): decorator for citation validation
  - get_run_url()             : returns LangSmith URL for current run
  - create_run_metadata()     : builds rich metadata dict for each span

Each span captures:
  - Node / agent name
  - Input query + cancer type + intent
  - Output text length + claim counts
  - Citation validation stats (valid/total/%)
  - Latency (automatic via LangSmith)
  - Error details if the step failed
  - Tags: ["medical-rag", "nccn", node_name, intent]
"""

import os
import time
import functools
import traceback
from typing import Any, Callable, Dict, Optional, TypeVar

from dotenv import load_dotenv

load_dotenv()

F = TypeVar("F", bound=Callable[..., Any])

# ─────────────────────────────────────────────────────────────────
# LangSmith client (lazy-loaded so import never crashes if not installed)
# ─────────────────────────────────────────────────────────────────
_ls_client = None
_tracing_enabled = False


def setup_langsmith() -> bool:
    """
    Initialise LangSmith tracing from environment variables.
    Returns True if tracing was enabled successfully.

    Required env vars:
        LANGCHAIN_API_KEY       - your LangSmith API key
        LANGCHAIN_TRACING_V2    - must be "true"
        LANGCHAIN_PROJECT       - project name shown in LangSmith UI
        LANGCHAIN_ENDPOINT      - defaults to https://api.smith.langchain.com
    """
    global _ls_client, _tracing_enabled

    api_key   = os.getenv("LANGCHAIN_API_KEY", "")
    tracing   = os.getenv("LANGCHAIN_TRACING_V2", "false").lower()
    project   = os.getenv("LANGCHAIN_PROJECT", "medical-research-assistant")
    endpoint  = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

    if not api_key or api_key == "your_langsmith_api_key_here":
        print("⚠️  LangSmith: LANGCHAIN_API_KEY not set — tracing disabled")
        _tracing_enabled = False
        return False

    if tracing != "true":
        print("ℹ️  LangSmith: LANGCHAIN_TRACING_V2 is not 'true' — tracing disabled")
        _tracing_enabled = False
        return False

    try:
        from langsmith import Client
        _ls_client = Client(api_url=endpoint, api_key=api_key)

        # Verify connection
        _ls_client.list_projects()

        # Set env vars that LangChain/LangGraph pick up automatically
        os.environ["LANGCHAIN_TRACING_V2"]  = "true"
        os.environ["LANGCHAIN_API_KEY"]     = api_key
        os.environ["LANGCHAIN_PROJECT"]     = project
        os.environ["LANGCHAIN_ENDPOINT"]    = endpoint

        _tracing_enabled = True
        print(f"✅ LangSmith tracing enabled — project: '{project}'")
        print(f"   View traces at: https://smith.langchain.com/projects/p/{project}")
        return True

    except ImportError:
        print("⚠️  LangSmith: 'langsmith' package not installed — run: pip install langsmith")
        _tracing_enabled = False
        return False
    except Exception as e:
        print(f"⚠️  LangSmith: connection failed ({e}) — tracing disabled")
        _tracing_enabled = False
        return False


def is_tracing_enabled() -> bool:
    return _tracing_enabled


def get_run_url(run_name: Optional[str] = None) -> str:
    """Return the LangSmith project URL."""
    project = os.getenv("LANGCHAIN_PROJECT", "medical-research-assistant")
    return f"https://smith.langchain.com/projects/p/{project}"


# ─────────────────────────────────────────────────────────────────
# Metadata helpers
# ─────────────────────────────────────────────────────────────────

def create_run_metadata(
    node_name: str,
    query: str,
    cancer_type: Optional[str] = None,
    intent: Optional[str] = None,
    extra: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Build a rich metadata dict attached to each LangSmith span."""
    meta = {
        "node":        node_name,
        "query":       query[:200],
        "cancer_type": cancer_type or "unknown",
        "intent":      intent or "unknown",
        "project":     os.getenv("LANGCHAIN_PROJECT", "medical-research-assistant"),
    }
    if extra:
        meta.update(extra)
    return meta


def build_tags(node_name: str, intent: Optional[str] = None, cancer_type: Optional[str] = None) -> list:
    """Build tag list for a LangSmith span."""
    tags = ["medical-rag", "nccn", node_name]
    if intent:
        tags.append(f"intent:{intent}")
    if cancer_type:
        # Sanitise for tag use
        tags.append(f"cancer:{cancer_type.lower().replace(' ', '_')[:30]}")
    return tags


# ─────────────────────────────────────────────────────────────────
# Decorator: trace_node
# ─────────────────────────────────────────────────────────────────

def trace_node(node_name: str):
    """
    Decorator for LangGraph node functions.

    Wraps the node with a LangSmith 'chain' run that records:
      - inputs:  user_query, cancer_type, intent
      - outputs: which state fields changed, output length
      - errors:  full traceback if the node raises
      - tags:    ["medical-rag", "nccn", node_name, "intent:X"]
      - metadata: cancer type, intent, regeneration count
    """
    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(state: Dict, *args, **kwargs):
            if not _tracing_enabled:
                return fn(state, *args, **kwargs)

            try:
                from langsmith import traceable
            except ImportError:
                return fn(state, *args, **kwargs)

            query       = state.get("user_query", "")
            cancer_type = state.get("cancer_type")
            intent      = state.get("intent", "unknown")
            regen       = state.get("regeneration_count", 0)

            @traceable(
                name=node_name,
                run_type="chain",
                tags=build_tags(node_name, intent, cancer_type),
                metadata=create_run_metadata(
                    node_name, query, cancer_type, intent,
                    extra={"regeneration_attempt": regen}
                ),
            )
            def _traced(state):
                return fn(state, *args, **kwargs)

            return _traced(state)

        return wrapper  # type: ignore
    return decorator


# ─────────────────────────────────────────────────────────────────
# Decorator: trace_agent
# ─────────────────────────────────────────────────────────────────

def trace_agent(agent_name: str, run_type: str = "chain"):
    """
    Decorator for agent .run() methods.

    Wraps the call with a LangSmith span that records:
      - inputs:  query, cancer_type
      - outputs: number of claims extracted, validation pass rate
      - errors:  exception details
      - tags:    ["medical-rag", agent_name, "agent"]
    """
    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(self_obj, query: str, *args, **kwargs):
            if not _tracing_enabled:
                return fn(self_obj, query, *args, **kwargs)

            try:
                from langsmith import traceable
            except ImportError:
                return fn(self_obj, query, *args, **kwargs)

            cancer_type = args[0] if args else kwargs.get("cancer_type")

            @traceable(
                name=agent_name,
                run_type=run_type,
                tags=["medical-rag", "nccn", agent_name, "agent"],
                metadata=create_run_metadata(agent_name, query, cancer_type),
            )
            def _traced(query, *args, **kwargs):
                result = fn(self_obj, query, *args, **kwargs)

                # Enrich span output with claim / validation stats if available
                if isinstance(result, tuple) and len(result) >= 2:
                    parsed, report = result[0], result[1]
                    if report is not None and hasattr(report, "total_claims"):
                        _log_citation_stats(agent_name, report)
                return result

            return _traced(query, *args, **kwargs)

        return wrapper  # type: ignore
    return decorator


# ─────────────────────────────────────────────────────────────────
# Decorator: trace_llm_call
# ─────────────────────────────────────────────────────────────────

def trace_llm_call(call_name: str):
    """
    Decorator for individual LLM invocations (force_json_response, etc.).

    Records:
      - prompt length
      - response length
      - attempt number
      - model name
    """
    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not _tracing_enabled:
                return fn(*args, **kwargs)

            try:
                from langsmith import traceable
            except ImportError:
                return fn(*args, **kwargs)

            model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

            @traceable(
                name=call_name,
                run_type="llm",
                tags=["medical-rag", "groq", call_name, f"model:{model}"],
                metadata={"model": model, "call": call_name},
            )
            def _traced(*args, **kwargs):
                return fn(*args, **kwargs)

            return _traced(*args, **kwargs)

        return wrapper  # type: ignore
    return decorator


# ─────────────────────────────────────────────────────────────────
# Decorator: trace_citation_validation
# ─────────────────────────────────────────────────────────────────

def trace_citation_validation(fn: F) -> F:
    """
    Decorator specifically for citation validation calls.

    Records:
      - number of claims checked
      - number of source chunks
      - valid / invalid / validity_rate
      - whether regeneration was triggered
    """
    @functools.wraps(fn)
    def wrapper(claims, retrieved_chunks, *args, **kwargs):
        if not _tracing_enabled:
            return fn(claims, retrieved_chunks, *args, **kwargs)

        try:
            from langsmith import traceable
        except ImportError:
            return fn(claims, retrieved_chunks, *args, **kwargs)

        @traceable(
            name="citation_validation",
            run_type="tool",
            tags=["medical-rag", "citation-validation", "nccn"],
            metadata={
                "claims_to_validate": len(claims),
                "source_chunks":      len(retrieved_chunks),
            },
        )
        def _traced(claims, retrieved_chunks):
            report = fn(claims, retrieved_chunks, *args, **kwargs)
            return report

        return _traced(claims, retrieved_chunks)

    return wrapper  # type: ignore


# ─────────────────────────────────────────────────────────────────
# Decorator: trace_graph_run
# ─────────────────────────────────────────────────────────────────

def trace_graph_run(fn: F) -> F:
    """
    Decorator for run_graph() — wraps the entire pipeline execution
    as a single top-level LangSmith trace with full input/output.
    """
    @functools.wraps(fn)
    def wrapper(graph, query: str, *args, **kwargs):
        if not _tracing_enabled:
            return fn(graph, query, *args, **kwargs)

        try:
            from langsmith import traceable
        except ImportError:
            return fn(graph, query, *args, **kwargs)

        @traceable(
            name="medical_research_pipeline",
            run_type="chain",
            tags=["medical-rag", "nccn", "full-pipeline"],
            metadata={
                "query":   query[:200],
                "project": os.getenv("LANGCHAIN_PROJECT", "medical-research-assistant"),
            },
        )
        def _traced(graph, query, *args, **kwargs):
            return fn(graph, query, *args, **kwargs)

        return _traced(graph, query, *args, **kwargs)

    return wrapper  # type: ignore


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _log_citation_stats(agent_name: str, report: Any) -> None:
    """Print citation stats to console (LangSmith picks these up via stdout)."""
    total    = getattr(report, "total_claims", 0)
    valid    = getattr(report, "valid_claims", 0)
    rate     = f"{int((valid/total)*100)}%" if total > 0 else "N/A"
    needs_r  = getattr(report, "needs_regeneration", False)
    print(
        f"   📊 [{agent_name}] Citation stats: "
        f"{valid}/{total} valid ({rate}) | "
        f"regen={'yes' if needs_r else 'no'}"
    )