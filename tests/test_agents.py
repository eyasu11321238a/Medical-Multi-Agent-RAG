"""
test_agents.py
--------------
Tests for the Medical Research Assistant agents.
Run: python -m pytest tests/test_agents.py -v
"""

import pytest
from unittest.mock import MagicMock, patch
from src.utils.helpers import (
    detect_cancer_type_from_query,
    detect_intent,
    detect_comparison_query,
    detect_all_cancer_types,
    format_final_response,
    add_medical_disclaimer,
)


# ─────────────────────────────────────────────
# Tests: Cancer Type Detection
# ─────────────────────────────────────────────

class TestCancerTypeDetection:

    def test_detects_basal_cell(self):
        query = "What are symptoms of basal cell carcinoma?"
        result = detect_cancer_type_from_query(query)
        assert result == "Basal Cell Skin Cancer"

    def test_detects_melanoma(self):
        query = "How is melanoma diagnosed?"
        result = detect_cancer_type_from_query(query)
        assert result == "Melanoma"

    def test_detects_breast_cancer(self):
        query = "What are treatment options for breast cancer?"
        result = detect_cancer_type_from_query(query)
        assert result == "Breast Cancer"

    def test_detects_lung_cancer(self):
        query = "Tell me about lung cancer treatment"
        result = detect_cancer_type_from_query(query)
        assert result == "Lung Cancer"

    def test_returns_none_for_unknown(self):
        query = "What is the weather today?"
        result = detect_cancer_type_from_query(query)
        assert result is None

    def test_detects_bcc_abbreviation(self):
        query = "Is BCC curable?"
        result = detect_cancer_type_from_query(query)
        assert result == "Basal Cell Skin Cancer"


# ─────────────────────────────────────────────
# Tests: Intent Detection
# ─────────────────────────────────────────────

class TestIntentDetection:

    def test_diagnosis_intent(self):
        query = "What are the symptoms of melanoma?"
        result = detect_intent(query)
        assert result == "diagnosis"

    def test_treatment_intent(self):
        query = "What surgery options are available for skin cancer?"
        result = detect_intent(query)
        assert result == "treatment"

    def test_summarization_intent(self):
        query = "Summarize the basal cell cancer guidelines"
        result = detect_intent(query)
        assert result == "summarization"

    def test_general_intent_default(self):
        query = "Tell me about skin cancer"
        result = detect_intent(query)
        assert result in ["summarization", "general"]


# ─────────────────────────────────────────────
# Tests: Comparison Detection
# ─────────────────────────────────────────────

class TestComparisonDetection:

    def test_detects_vs_comparison(self):
        query = "melanoma vs basal cell cancer treatment"
        result = detect_comparison_query(query)
        assert result is True

    def test_detects_compare_keyword(self):
        query = "compare breast cancer and lung cancer"
        result = detect_comparison_query(query)
        assert result is True

    def test_no_comparison_in_simple_query(self):
        query = "What is melanoma?"
        result = detect_comparison_query(query)
        assert result is False

    def test_detects_multiple_cancer_types(self):
        query = "compare melanoma vs basal cell skin cancer"
        result = detect_all_cancer_types(query)
        assert "Melanoma" in result
        assert "Basal Cell Skin Cancer" in result
        assert len(result) >= 2


# ─────────────────────────────────────────────
# Tests: Response Formatting
# ─────────────────────────────────────────────

class TestResponseFormatting:

    def test_format_diagnosis_response(self):
        result = format_final_response(
            intent="diagnosis",
            cancer_type="Melanoma",
            diagnosis_output="Melanoma is a serious skin cancer."
        )
        assert "Melanoma" in result
        assert "Diagnosis" in result

    def test_format_treatment_response(self):
        result = format_final_response(
            intent="treatment",
            cancer_type="Breast Cancer",
            treatment_output="Surgery and radiation are common."
        )
        assert "Treatment" in result

    def test_disclaimer_added(self):
        response = "This is a test response."
        result = add_medical_disclaimer(response)
        assert "Medical Disclaimer" in result
        assert "healthcare provider" in result

    def test_format_with_no_output_returns_message(self):
        result = format_final_response(
            intent="general",
            cancer_type=None
        )
        assert "could not find" in result.lower()


# ─────────────────────────────────────────────
# Tests: Graph State Routing
# ─────────────────────────────────────────────

class TestGraphRouting:
    """Test the routing logic without calling LLM."""

    def test_diagnosis_query_routes_correctly(self):
        from src.graph.medical_graph import route_after_supervisor
        state = {
            "user_query": "What are symptoms of melanoma?",
            "cancer_type": "Melanoma",
            "intent": "diagnosis",
            "is_comparison": False,
            "cancer_types_detected": ["Melanoma"],
            "diagnosis_output": None,
            "treatment_output": None,
            "summary_output": None,
            "final_response": None,
            "messages": [],
            "error": None,
        }
        result = route_after_supervisor(state)
        assert result == "diagnosis"

    def test_treatment_query_routes_correctly(self):
        from src.graph.medical_graph import route_after_supervisor
        state = {
            "user_query": "What surgery is used for basal cell?",
            "cancer_type": "Basal Cell Skin Cancer",
            "intent": "treatment",
            "is_comparison": False,
            "cancer_types_detected": ["Basal Cell Skin Cancer"],
            "diagnosis_output": None,
            "treatment_output": None,
            "summary_output": None,
            "final_response": None,
            "messages": [],
            "error": None,
        }
        result = route_after_supervisor(state)
        assert result == "treatment"

    def test_comparison_routes_to_summarization(self):
        from src.graph.medical_graph import route_after_supervisor
        state = {
            "user_query": "Compare melanoma vs basal cell cancer",
            "cancer_type": None,
            "intent": "summarization",
            "is_comparison": True,
            "cancer_types_detected": ["Melanoma", "Basal Cell Skin Cancer"],
            "diagnosis_output": None,
            "treatment_output": None,
            "summary_output": None,
            "final_response": None,
            "messages": [],
            "error": None,
        }
        result = route_after_supervisor(state)
        assert result == "summarization"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
