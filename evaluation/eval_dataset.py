"""
eval_dataset.py
---------------
20 NCCN-based evaluation questions with:
  - structured expected answers (key concepts that must appear)
  - expected citation pages / chapters
  - category tags for grouped analysis
  - difficulty level

Categories:
  DIAGNOSIS   - signs, symptoms, risk factors, staging
  TREATMENT   - surgical and non-surgical options
  SYSTEMIC    - drugs, immunotherapy, targeted therapy
  FOLLOW_UP   - monitoring and surveillance
  COMPARISON  - cross-cancer comparisons
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EvalQuestion:
    id:                  int
    question:            str
    category:            str          # DIAGNOSIS | TREATMENT | SYSTEMIC | FOLLOW_UP | COMPARISON
    cancer_type:         str
    difficulty:          str          # easy | medium | hard
    expected_concepts:   List[str]    # key terms/phrases that MUST appear in a good answer
    expected_citations:  List[dict]   # list of {"chapter": str, "keywords": List[str]}
    notes:               str = ""     # why this question tests a specific capability


# ─────────────────────────────────────────────────────────────────
# THE 20 EVALUATION QUESTIONS
# ─────────────────────────────────────────────────────────────────

EVAL_DATASET: List[EvalQuestion] = [

    # ── DIAGNOSIS (Q1–Q5) ─────────────────────────────────────────

    EvalQuestion(
        id=1,
        question="What are the main signs and symptoms of basal cell carcinoma (BCC)?",
        category="DIAGNOSIS",
        cancer_type="Basal Cell Skin Cancer",
        difficulty="easy",
        expected_concepts=[
            "pearly", "waxy", "rolled border", "telangiectasia",
            "ulceration", "nodular", "pigmented", "sclerosing",
        ],
        expected_citations=[
            {"chapter": "Signs and Symptoms", "keywords": ["pearly", "lesion", "basal cell"]},
            {"chapter": "Clinical Presentation", "keywords": ["nodular", "superficial"]},
        ],
        notes="Tests basic symptom recognition — should be well-covered in any NCCN PDF.",
    ),

    EvalQuestion(
        id=2,
        question="What risk factors increase the likelihood of developing basal cell carcinoma?",
        category="DIAGNOSIS",
        cancer_type="Basal Cell Skin Cancer",
        difficulty="easy",
        expected_concepts=[
            "UV exposure", "sun exposure", "fair skin", "immunosuppression",
            "radiation therapy history", "arsenic", "Gorlin syndrome",
        ],
        expected_citations=[
            {"chapter": "Risk Factors", "keywords": ["UV", "sun", "skin type"]},
        ],
        notes="Tests risk factor coverage — multiple factors should be cited with pages.",
    ),

    EvalQuestion(
        id=3,
        question="How is basal cell carcinoma staged or risk-classified according to NCCN guidelines?",
        category="DIAGNOSIS",
        cancer_type="Basal Cell Skin Cancer",
        difficulty="medium",
        expected_concepts=[
            "low risk", "high risk", "location", "size", "perineural invasion",
            "poorly defined borders", "recurrent", "immunosuppressed",
        ],
        expected_citations=[
            {"chapter": "Risk Classification", "keywords": ["low risk", "high risk", "BCC"]},
            {"chapter": "Staging", "keywords": ["tumor", "size", "location"]},
        ],
        notes="Tests structured risk classification — key decision point in NCCN guidelines.",
    ),

    EvalQuestion(
        id=4,
        question="What are the clinical features that distinguish high-risk from low-risk BCC?",
        category="DIAGNOSIS",
        cancer_type="Basal Cell Skin Cancer",
        difficulty="medium",
        expected_concepts=[
            "location H-zone", "size >2cm", "poorly defined borders",
            "perineural invasion", "recurrence", "aggressive histology",
        ],
        expected_citations=[
            {"chapter": "Risk Stratification", "keywords": ["H-zone", "size", "high risk"]},
        ],
        notes="Tests ability to contrast risk tiers — requires precise citation of criteria.",
    ),

    EvalQuestion(
        id=5,
        question="What diagnostic workup is recommended for suspected basal cell carcinoma?",
        category="DIAGNOSIS",
        cancer_type="Basal Cell Skin Cancer",
        difficulty="easy",
        expected_concepts=[
            "biopsy", "shave biopsy", "punch biopsy", "histopathology",
            "dermoscopy", "pathological confirmation",
        ],
        expected_citations=[
            {"chapter": "Diagnosis", "keywords": ["biopsy", "pathology", "diagnosis"]},
        ],
        notes="Tests diagnostic workup knowledge — standard NCCN recommendation.",
    ),

    # ── TREATMENT (Q6–Q12) ────────────────────────────────────────

    EvalQuestion(
        id=6,
        question="What surgical options exist for low-risk basal cell carcinoma?",
        category="TREATMENT",
        cancer_type="Basal Cell Skin Cancer",
        difficulty="easy",
        expected_concepts=[
            "standard excision", "curettage", "electrodessication",
            "cryotherapy", "surgical margins", "1cm margin",
        ],
        expected_citations=[
            {"chapter": "Treatment", "keywords": ["excision", "low risk", "surgery"]},
            {"chapter": "Surgical Options", "keywords": ["curettage", "electrodessication"]},
        ],
        notes="Tests basic surgical knowledge — clear NCCN recommendation for low-risk.",
    ),

    EvalQuestion(
        id=7,
        question="What surgery options exist for high-risk BCC?",
        category="TREATMENT",
        cancer_type="Basal Cell Skin Cancer",
        difficulty="medium",
        expected_concepts=[
            "Mohs surgery", "Mohs micrographic surgery", "complete excision",
            "peripheral and deep en face margin assessment", "PDEMA",
            "wide excision",
        ],
        expected_citations=[
            {"chapter": "Treatment", "keywords": ["Mohs", "high risk", "surgery"]},
            {"chapter": "Surgical Excision", "keywords": ["margin", "high risk"]},
        ],
        notes="The exact question the user tested — must return cited Mohs and excision details.",
    ),

    EvalQuestion(
        id=8,
        question="When is Mohs micrographic surgery preferred over standard excision for BCC?",
        category="TREATMENT",
        cancer_type="Basal Cell Skin Cancer",
        difficulty="medium",
        expected_concepts=[
            "high risk", "H-zone", "recurrent BCC", "cosmetically sensitive area",
            "ill-defined borders", "perineural invasion", "tissue sparing",
        ],
        expected_citations=[
            {"chapter": "Mohs Surgery", "keywords": ["Mohs", "indication", "preferred"]},
        ],
        notes="Tests Mohs indications — requires precise clinical criteria from guidelines.",
    ),

    EvalQuestion(
        id=9,
        question="What non-surgical treatment options are available for BCC?",
        category="TREATMENT",
        cancer_type="Basal Cell Skin Cancer",
        difficulty="medium",
        expected_concepts=[
            "radiation therapy", "imiquimod", "5-fluorouracil", "5-FU",
            "photodynamic therapy", "PDT", "topical therapy",
        ],
        expected_citations=[
            {"chapter": "Non-Surgical Treatment", "keywords": ["radiation", "topical", "BCC"]},
            {"chapter": "Treatment Options", "keywords": ["imiquimod", "5-FU"]},
        ],
        notes="Tests non-surgical alternatives — important for inoperable or elderly patients.",
    ),

    EvalQuestion(
        id=10,
        question="What systemic therapies are recommended for advanced or metastatic BCC?",
        category="SYSTEMIC",
        cancer_type="Basal Cell Skin Cancer",
        difficulty="hard",
        expected_concepts=[
            "vismodegib", "sonidegib", "hedgehog pathway inhibitor",
            "cemiplimab", "PD-1 inhibitor", "checkpoint inhibitor",
            "locally advanced", "metastatic",
        ],
        expected_citations=[
            {"chapter": "Systemic Therapy", "keywords": ["vismodegib", "advanced BCC"]},
            {"chapter": "Hedgehog Inhibitor", "keywords": ["sonidegib", "locally advanced"]},
        ],
        notes="Tests knowledge of hedgehog inhibitors — key NCCN recommendation for advanced BCC.",
    ),

    EvalQuestion(
        id=11,
        question="What is the recommended treatment for superficial BCC on the trunk?",
        category="TREATMENT",
        cancer_type="Basal Cell Skin Cancer",
        difficulty="medium",
        expected_concepts=[
            "curettage", "electrodessication", "imiquimod", "5-FU",
            "photodynamic therapy", "low risk", "superficial",
        ],
        expected_citations=[
            {"chapter": "Superficial BCC", "keywords": ["superficial", "trunk", "treatment"]},
        ],
        notes="Tests site-specific treatment — trunk is low-risk H-zone exempt.",
    ),

    EvalQuestion(
        id=12,
        question="How should BCC near the eye or nose be treated?",
        category="TREATMENT",
        cancer_type="Basal Cell Skin Cancer",
        difficulty="hard",
        expected_concepts=[
            "H-zone", "Mohs surgery", "multidisciplinary", "ophthalmology",
            "periorbital", "high risk location", "tissue sparing",
        ],
        expected_citations=[
            {"chapter": "H-Zone Treatment", "keywords": ["H-zone", "eye", "nose", "Mohs"]},
        ],
        notes="Tests H-zone awareness — critical for face/eye/nose BCC management.",
    ),

    # ── SYSTEMIC THERAPY (Q13–Q14) ────────────────────────────────

    EvalQuestion(
        id=13,
        question="What is vismodegib and when is it indicated for BCC?",
        category="SYSTEMIC",
        cancer_type="Basal Cell Skin Cancer",
        difficulty="hard",
        expected_concepts=[
            "vismodegib", "Erivedge", "hedgehog pathway", "Smoothened inhibitor",
            "locally advanced BCC", "metastatic BCC", "inoperable",
        ],
        expected_citations=[
            {"chapter": "Systemic Therapy", "keywords": ["vismodegib", "hedgehog", "advanced"]},
        ],
        notes="Tests specific drug knowledge — FDA-approved targeted therapy for BCC.",
    ),

    EvalQuestion(
        id=14,
        question="What immunotherapy options exist for locally advanced BCC that progressed on hedgehog inhibitors?",
        category="SYSTEMIC",
        cancer_type="Basal Cell Skin Cancer",
        difficulty="hard",
        expected_concepts=[
            "cemiplimab", "PD-1", "checkpoint inhibitor", "immunotherapy",
            "progression", "hedgehog inhibitor", "second-line",
        ],
        expected_citations=[
            {"chapter": "Immunotherapy", "keywords": ["cemiplimab", "PD-1", "BCC"]},
        ],
        notes="Tests second-line therapy knowledge — cemiplimab is the key NCCN recommendation.",
    ),

    # ── FOLLOW-UP (Q15–Q16) ───────────────────────────────────────

    EvalQuestion(
        id=15,
        question="What follow-up schedule is recommended after BCC treatment?",
        category="FOLLOW_UP",
        cancer_type="Basal Cell Skin Cancer",
        difficulty="medium",
        expected_concepts=[
            "6 months", "annual", "skin examination", "lymph node",
            "recurrence", "new primary", "sun protection",
        ],
        expected_citations=[
            {"chapter": "Follow-up", "keywords": ["surveillance", "follow-up", "BCC"]},
            {"chapter": "Monitoring", "keywords": ["skin exam", "annual", "recurrence"]},
        ],
        notes="Tests surveillance knowledge — important for patient management.",
    ),

    EvalQuestion(
        id=16,
        question="What patient education should be provided after BCC treatment?",
        category="FOLLOW_UP",
        cancer_type="Basal Cell Skin Cancer",
        difficulty="easy",
        expected_concepts=[
            "sun protection", "sunscreen", "SPF", "protective clothing",
            "self-examination", "avoid tanning", "follow-up visits",
        ],
        expected_citations=[
            {"chapter": "Patient Education", "keywords": ["sun protection", "sunscreen"]},
        ],
        notes="Tests patient education content — practical NCCN guidance.",
    ),

    # ── COMPARISON (Q17–Q20) ──────────────────────────────────────

    EvalQuestion(
        id=17,
        question="How do BCC and squamous cell carcinoma (SCC) differ in terms of metastatic risk?",
        category="COMPARISON",
        cancer_type="Basal Cell Skin Cancer",
        difficulty="hard",
        expected_concepts=[
            "BCC rarely metastasizes", "SCC higher metastatic risk",
            "lymph node", "perineural invasion", "regional spread",
        ],
        expected_citations=[
            {"chapter": "Metastasis", "keywords": ["metastatic", "BCC", "SCC", "rare"]},
        ],
        notes="Tests comparative oncology knowledge across cancer types.",
    ),

    EvalQuestion(
        id=18,
        question="How does the treatment of recurrent BCC differ from primary BCC?",
        category="TREATMENT",
        cancer_type="Basal Cell Skin Cancer",
        difficulty="hard",
        expected_concepts=[
            "Mohs surgery preferred", "wider margins", "recurrent",
            "aggressive subtype", "re-excision", "radiation therapy",
        ],
        expected_citations=[
            {"chapter": "Recurrent BCC", "keywords": ["recurrent", "re-excision", "Mohs"]},
        ],
        notes="Tests recurrence management — distinct NCCN pathway for recurrent disease.",
    ),

    EvalQuestion(
        id=19,
        question="What is the role of radiation therapy in BCC management?",
        category="TREATMENT",
        cancer_type="Basal Cell Skin Cancer",
        difficulty="medium",
        expected_concepts=[
            "radiation therapy", "RT", "inoperable", "adjuvant",
            "elderly patients", "perineural invasion", "positive margins",
        ],
        expected_citations=[
            {"chapter": "Radiation Therapy", "keywords": ["radiation", "RT", "BCC"]},
        ],
        notes="Tests radiation therapy indications — often an alternative to surgery.",
    ),

    EvalQuestion(
        id=20,
        question="What are the NCCN recommended margins for surgical excision of BCC?",
        category="TREATMENT",
        cancer_type="Basal Cell Skin Cancer",
        difficulty="medium",
        expected_concepts=[
            "4mm margin", "low risk", "high risk",
            "complete margin assessment", "standard excision",
        ],
        expected_citations=[
            {"chapter": "Surgical Margins", "keywords": ["margin", "excision", "BCC"]},
            {"chapter": "Treatment Guidelines", "keywords": ["4mm", "surgical margin"]},
        ],
        notes="Tests precise margin knowledge — numeric values expected in answer.",
    ),
]


def get_questions_by_category(category: str) -> List[EvalQuestion]:
    return [q for q in EVAL_DATASET if q.category == category]


def get_questions_by_difficulty(difficulty: str) -> List[EvalQuestion]:
    return [q for q in EVAL_DATASET if q.difficulty == difficulty]


def summary():
    cats = {}
    diffs = {}
    for q in EVAL_DATASET:
        cats[q.category]   = cats.get(q.category, 0) + 1
        diffs[q.difficulty] = diffs.get(q.difficulty, 0) + 1
    print(f"Total questions : {len(EVAL_DATASET)}")
    print(f"By category     : {cats}")
    print(f"By difficulty   : {diffs}")


if __name__ == "__main__":
    summary()
