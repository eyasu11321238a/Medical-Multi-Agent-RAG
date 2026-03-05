"""
citation_validator.py
---------------------
Validates that every cited claim actually exists in the retrieved chunks.
Checks page number, source file, and quote similarity.
Rejects or flags claims where citations cannot be verified.
"""

import re
import os
from typing import List, Tuple, Optional
from difflib import SequenceMatcher

from langchain_core.documents import Document
from dotenv import load_dotenv

from src.utils.structured_output import (
    Claim, Citation, CitationValidationResult, ValidationReport
)

load_dotenv()

QUOTE_MATCH_THRESHOLD     = 0.35
PAGE_MATCH_BONUS          = 0.20
MIN_VALIDITY_RATE         = 0.30
MAX_REGENERATION_ATTEMPTS = 2


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def _find_best_chunk_match(
    citation: Citation,
    chunks: List[Document],
) -> Tuple[float, Optional[Document], str]:
    if not chunks:
        return 0.0, None, "No source chunks available for validation"

    file_filtered = [
        c for c in chunks
        if citation.source_file.lower() in
           c.metadata.get("source_file", "").lower()
    ]
    search_pool  = file_filtered if file_filtered else chunks
    file_matched = bool(file_filtered)

    page_filtered = [
        c for c in search_pool
        if c.metadata.get("page_number") == citation.page_number
    ]
    page_matched = bool(page_filtered)
    if page_filtered:
        search_pool = page_filtered

    best_score = 0.0
    best_chunk = None

    for chunk in search_pool:
        score = _similarity(citation.quote, chunk.page_content)
        if page_matched:
            score = min(1.0, score + PAGE_MATCH_BONUS)
        if score > best_score:
            best_score = score
            best_chunk = chunk

    failure_reason = None
    if best_score < QUOTE_MATCH_THRESHOLD:
        parts = []
        if not file_matched:
            parts.append(f"source file '{citation.source_file}' not in retrieved chunks")
        if not page_matched:
            parts.append(f"page {citation.page_number} not in retrieved chunks")
        parts.append(f"quote similarity {best_score:.2f} below threshold {QUOTE_MATCH_THRESHOLD}")
        failure_reason = "; ".join(parts)

    return best_score, best_chunk, failure_reason


def validate_claims(
    claims: List[Claim],
    retrieved_chunks: List[Document],
) -> ValidationReport:
    results: List[CitationValidationResult] = []
    valid_count = 0

    print(f"\n   Validating {len(claims)} claims against {len(retrieved_chunks)} source chunks...")

    for claim in claims:
        score, matched_chunk, failure_reason = _find_best_chunk_match(
            claim.citation, retrieved_chunks
        )
        is_valid = score >= QUOTE_MATCH_THRESHOLD

        if is_valid:
            valid_count += 1
            print(f"      OK  Claim {claim.claim_id}: valid (score={score:.2f})")
        else:
            print(f"      FAIL Claim {claim.claim_id}: INVALID - {failure_reason}")

        results.append(CitationValidationResult(
            claim_id=claim.claim_id,
            statement=claim.statement,
            citation=claim.citation,
            is_valid=is_valid,
            match_score=round(score, 3),
            failure_reason=failure_reason,
        ))

    total         = len(claims)
    invalid       = total - valid_count
    validity_rate = valid_count / total if total > 0 else 0.0
    needs_regen   = validity_rate < MIN_VALIDITY_RATE

    failure_summary = None
    if needs_regen:
        failure_summary = (
            f"{invalid}/{total} claims failed citation validation "
            f"(validity rate {validity_rate:.0%} < required {MIN_VALIDITY_RATE:.0%}). "
            f"Regeneration required."
        )
        print(f"   WARNING: {failure_summary}")
    else:
        print(f"   PASS: {valid_count}/{total} claims verified (rate={validity_rate:.0%})")

    return ValidationReport(
        total_claims=total,
        valid_claims=valid_count,
        invalid_claims=invalid,
        validation_results=results,
        overall_valid=not needs_regen,
        needs_regeneration=needs_regen,
        failure_summary=failure_summary,
    )


def filter_to_valid_claims(
    claims: List[Claim],
    report: ValidationReport,
) -> List[Claim]:
    valid_ids = {r.claim_id for r in report.validation_results if r.is_valid}
    return [c for c in claims if c.claim_id in valid_ids]


def format_validated_response(
    claims: List[Claim],
    report: ValidationReport,
    summary: str,
    cancer_type: Optional[str] = None,
) -> str:
    """
    Format validated claims into rich markdown with full citation blocks.
    Each verified claim shows: source PDF filename, page number, chapter, and quote.
    Unverified claims are flagged separately.
    """
    valid_ids   = {r.claim_id for r in report.validation_results if r.is_valid}
    invalid_ids = {r.claim_id for r in report.validation_results if not r.is_valid}

    valid_claims   = [c for c in claims if c.claim_id in valid_ids]
    invalid_claims = [c for c in claims if c.claim_id in invalid_ids]

    lines = []

    if cancer_type:
        lines.append(f"### {cancer_type}\n")

    # Verified claims with full citation blocks
    if valid_claims:
        lines.append("### Verified Findings\n")
        for claim in valid_claims:
            cit       = claim.citation
            conf_icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(claim.confidence, "⚪")

            quote_display = cit.quote.strip()
            if len(quote_display) > 120:
                quote_display = quote_display[:120] + "..."

            src_file = cit.source_file
            pg       = cit.page_number
            chapter  = cit.chapter

            lines.append(
                f"{conf_icon} **{claim.statement}**\n"
                f"\n"
                f"> **Source file:** `{src_file}`\n"
                f"> **Page number:** {pg}\n"
                f"> **Chapter:** {chapter}\n"
                f"> **Supporting quote:** *\"{quote_display}\"*\n"
            )

    # Unverified claims
    if invalid_claims:
        lines.append("\n---\n### Unverified Claims\n")
        lines.append("*The following claims could not be matched to a source page:*\n")
        for claim in invalid_claims:
            lines.append(f"- {claim.statement} `[citation not verified]`\n")

    # Summary
    if summary:
        lines.append(f"\n---\n**Summary:** {summary}")

    # Citation stats footer
    total = report.total_claims
    valid = report.valid_claims
    if total > 0:
        pct = int((valid / total) * 100)
        lines.append(
            f"\n\n---\n"
            f"*Citation verification: **{valid}/{total}** claims verified ({pct}%)*"
        )

    return "\n".join(lines)