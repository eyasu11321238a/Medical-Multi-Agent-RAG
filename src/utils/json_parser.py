"""
json_parser.py
--------------
Forces structured JSON output. Includes hard enforcement that claim
arrays must be populated — rejects responses where all arrays are empty.
"""

import re
import json
import os
from typing import Type, TypeVar, Optional, Tuple
from pydantic import BaseModel, ValidationError

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

T = TypeVar("T", bound=BaseModel)

MAX_PARSE_RETRIES = 2

# This prefix is injected into every agent system prompt
JSON_ENFORCEMENT_PREFIX = """

=== STRICT JSON OUTPUT RULES ===
You MUST respond with ONLY valid JSON — no markdown, no explanation, no ```fences.
Start with {{ and end with }}.

CRITICAL: You MUST populate the claim arrays (surgical_options, primary_treatments,
symptoms, key_findings, etc.). Do NOT leave them as empty [].
Each claim array entry MUST have this exact structure:
{{
  "claim_id": 1,
  "statement": "The factual statement here",
  "confidence": "high",
  "citation": {{
    "source_file": "nccn_basal_cell_2026.pdf",
    "page_number": 12,
    "cancer_type": "Basal Cell Skin Cancer",
    "chapter": "Treatment",
    "quote": "exact short quote from the context above"
  }}
}}

Use page_number and source_file from the [Source N] headers in the context.
The quote MUST be 5–30 words copied verbatim from the context.
Every claim MUST have a citation. No citation = claim is invalid.
=================================
"""

RETRY_PROMPT = """Your previous JSON response was rejected. Reason: {error}

You MUST fix this and respond with ONLY valid JSON.
CRITICAL: The claim arrays (surgical_options, primary_treatments, etc.) MUST have 
at least one entry each. Do NOT leave them empty.

Use this exact format for each claim:
{{
  "claim_id": 1,
  "statement": "factual statement",
  "confidence": "high",
  "citation": {{
    "source_file": "filename.pdf",
    "page_number": 5,
    "cancer_type": "Cancer Type",
    "chapter": "Chapter Name",
    "quote": "verbatim quote from context"
  }}
}}

Original question: {query}
Context (use page numbers and filenames from the [Source N] headers):
{context}
"""


def _extract_json(text: str) -> str:
    """Strip markdown fences and extract JSON object."""
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    return text[start:]


def _get_schema_str(model: Type[T]) -> str:
    try:
        schema = model.model_json_schema()
        return json.dumps(schema, indent=2)[:2000]
    except Exception:
        return str(model.__fields__)


def _has_populated_claims(data: dict) -> bool:
    """Return True if at least one claim array has entries."""
    claim_keys = [
        "surgical_options", "primary_treatments", "non_surgical",
        "systemic_therapy", "followup_care", "symptoms", "risk_factors",
        "diagnostic_tests", "overview", "key_findings", "comparison_points",
    ]
    for key in claim_keys:
        if key in data and isinstance(data[key], list) and len(data[key]) > 0:
            return True
    return False


def force_json_response(
    llm: ChatGroq,
    system_prompt: str,
    user_prompt: str,
    output_model: Type[T],
    query: str = "",
    context: str = "",
    max_retries: int = MAX_PARSE_RETRIES,
) -> Tuple[Optional[T], bool, str]:
    """
    Call LLM and force structured JSON matching a Pydantic model.
    Rejects responses where all claim arrays are empty.
    Returns (parsed_model, success, error_message).
    """
    schema_str       = _get_schema_str(output_model)
    enforced_system  = system_prompt + JSON_ENFORCEMENT_PREFIX
    messages = [
        SystemMessage(content=enforced_system),
        HumanMessage(content=user_prompt),
    ]

    last_error = ""

    for attempt in range(max_retries + 1):
        try:
            if attempt == 0:
                print(f"      JSON attempt {attempt + 1}/{max_retries + 1}...")
                response = llm.invoke(messages)
            else:
                print(f"      Retry {attempt}/{max_retries} (reason: {last_error[:80]})...")
                retry_messages = [
                    SystemMessage(content=enforced_system),
                    HumanMessage(content=RETRY_PROMPT.format(
                        error=last_error,
                        query=query,
                        context=context[:2000],
                    )),
                ]
                response = llm.invoke(retry_messages)

            raw_text = response.content
            json_str = _extract_json(raw_text)
            data     = json.loads(json_str)

            # Hard check: claims must be populated
            if not _has_populated_claims(data):
                last_error = (
                    "All claim arrays are empty. You MUST populate at least one of: "
                    "surgical_options, primary_treatments, symptoms, key_findings, etc. "
                    "Each entry must have claim_id, statement, confidence, and citation."
                )
                print(f"      REJECTED: empty claims — {last_error[:80]}")
                continue

            # Pydantic validation
            parsed = output_model.model_validate(data)
            print(f"      JSON parsed OK — {len(parsed.all_claims())} claims extracted")
            return parsed, True, ""

        except json.JSONDecodeError as e:
            last_error = f"JSON decode error: {e}. Raw: {raw_text[:200]}"
            print(f"      JSON error: {e}")
        except ValidationError as e:
            last_error = f"Schema validation error: {e}"
            print(f"      Pydantic error: {str(e)[:150]}")
        except Exception as e:
            last_error = f"Unexpected error: {e}"
            print(f"      Error: {e}")
            break

    print(f"      All attempts failed. Last error: {last_error[:100]}")
    return None, False, last_error