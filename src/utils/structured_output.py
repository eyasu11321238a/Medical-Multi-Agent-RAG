"""
structured_output.py
--------------------
Pydantic models for structured JSON output.
Claims arrays use min_length=1 to force the LLM to populate them.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field, model_validator


class Citation(BaseModel):
    source_file:  str = Field(..., description="PDF filename, e.g. nccn_basal_cell_2026.pdf")
    page_number:  int = Field(..., ge=1, description="Page number in the PDF (integer)")
    cancer_type:  str = Field(..., description="Cancer type this source covers")
    chapter:      str = Field(default="General", description="Chapter or section name")
    quote:        str = Field(..., description="Verbatim short quote (5-30 words) from the source")


class Claim(BaseModel):
    claim_id:   int  = Field(..., description="Sequential number starting from 1")
    statement:  str  = Field(..., description="The factual statement")
    confidence: Literal["high", "medium", "low"] = Field(...)
    citation:   Citation = Field(..., description="Mandatory citation for this claim")

    @model_validator(mode="after")
    def validate_statement(self):
        if not self.statement or len(self.statement.strip()) < 10:
            raise ValueError("Claim statement must be at least 10 characters")
        if not self.citation.quote or len(self.citation.quote.strip()) < 5:
            raise ValueError("Citation quote must be at least 5 characters")
        return self


class DiagnosisResponse(BaseModel):
    cancer_type:      str         = Field(...)
    risk_level:       Optional[str] = Field(None)
    symptoms:         List[Claim] = Field(default_factory=list)
    risk_factors:     List[Claim] = Field(default_factory=list)
    diagnostic_tests: List[Claim] = Field(default_factory=list)
    overview:         List[Claim] = Field(default_factory=list)
    summary:          str         = Field(..., description="2-3 sentence plain-language summary")

    def all_claims(self) -> List[Claim]:
        return self.overview + self.symptoms + self.risk_factors + self.diagnostic_tests


class TreatmentResponse(BaseModel):
    cancer_type:        str         = Field(...)
    risk_level:         Optional[str] = Field(None)
    primary_treatments: List[Claim] = Field(default_factory=list)
    surgical_options:   List[Claim] = Field(default_factory=list)
    non_surgical:       List[Claim] = Field(default_factory=list)
    systemic_therapy:   List[Claim] = Field(default_factory=list)
    followup_care:      List[Claim] = Field(default_factory=list)
    summary:            str         = Field(..., description="2-3 sentence plain-language summary")

    def all_claims(self) -> List[Claim]:
        return (self.primary_treatments + self.surgical_options +
                self.non_surgical + self.systemic_therapy + self.followup_care)


class SummarizationResponse(BaseModel):
    question_answered: str         = Field(...)
    key_findings:      List[Claim] = Field(default_factory=list)
    comparison_points: List[Claim] = Field(default_factory=list)
    summary:           str         = Field(...)

    def all_claims(self) -> List[Claim]:
        return self.key_findings + self.comparison_points


class MergedResponse(BaseModel):
    query:            str                             = Field(...)
    cancer_type:      Optional[str]                   = Field(None)
    intent:           str                             = Field(...)
    diagnosis_section: Optional[DiagnosisResponse]   = Field(None)
    treatment_section: Optional[TreatmentResponse]   = Field(None)
    summary_section:   Optional[SummarizationResponse] = Field(None)
    overall_summary:  str                             = Field(...)
    total_claims:     int                             = Field(default=0)
    verified_claims:  int                             = Field(default=0)
    confidence_score: int                             = Field(default=0, ge=0, le=100)

    def all_claims(self) -> List[Claim]:
        claims = []
        if self.diagnosis_section:
            claims += self.diagnosis_section.all_claims()
        if self.treatment_section:
            claims += self.treatment_section.all_claims()
        if self.summary_section:
            claims += self.summary_section.all_claims()
        return claims


class CitationValidationResult(BaseModel):
    claim_id:       int
    statement:      str
    citation:       Citation
    is_valid:       bool
    match_score:    float = Field(ge=0.0, le=1.0)
    failure_reason: Optional[str] = Field(None)


class ValidationReport(BaseModel):
    total_claims:       int
    valid_claims:       int
    invalid_claims:     int
    validation_results: List[CitationValidationResult]
    overall_valid:      bool
    needs_regeneration: bool
    failure_summary:    Optional[str] = None

    @property
    def validity_rate(self) -> float:
        if self.total_claims == 0:
            return 0.0
        return self.valid_claims / self.total_claims