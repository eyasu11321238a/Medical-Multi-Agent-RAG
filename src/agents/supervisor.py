"""
supervisor.py
-------------
Supervisor Agent powered by Groq. Orchestrates the multi-agent pipeline.
"""

import os
from typing import Optional, Dict, Any

from langchain_groq import ChatGroq
from dotenv import load_dotenv

from src.rag.vector_store import MedicalRetriever
from src.agents.diagnosis_agent import DiagnosisAgent
from src.agents.treatment_agent import TreatmentAgent
from src.agents.summarization_agent import SummarizationAgent
from src.utils.helpers import (
    detect_cancer_type_from_query,
    detect_intent,
    detect_comparison_query,
    detect_all_cancer_types,
    format_final_response,
    add_medical_disclaimer,
)

load_dotenv()


class SupervisorAgent:
    """
    Orchestrates the multi-agent pipeline:
    1. Analyzes incoming query (intent, cancer type, comparison)
    2. Routes to appropriate agent(s)
    3. Merges outputs if needed
    4. Returns final structured response
    """

    def __init__(self, retriever: MedicalRetriever):
        self.retriever = retriever
        self.diagnosis_agent      = DiagnosisAgent(retriever)
        self.treatment_agent      = TreatmentAgent(retriever)
        self.summarization_agent  = SummarizationAgent(retriever)
        self.llm = ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            temperature=0.0,
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )
        print("✅ Supervisor Agent initialized with all sub-agents (Groq)")

    def analyze_query(self, query: str) -> Dict[str, Any]:
        cancer_type   = detect_cancer_type_from_query(query)
        intent        = detect_intent(query)
        is_comparison = detect_comparison_query(query)
        all_cancers   = detect_all_cancer_types(query)

        agents = []
        if is_comparison and len(all_cancers) >= 2:
            agents = ["summarization_comparison"]
        elif intent == "diagnosis":
            agents = ["diagnosis"]
        elif intent == "treatment":
            agents = ["treatment"]
        elif intent in ("summarization", "general"):
            agents = ["summarization"]
        else:
            agents = ["diagnosis", "treatment"]

        plan = {
            "query": query,
            "cancer_type": cancer_type,
            "intent": intent,
            "is_comparison": is_comparison,
            "cancer_types_detected": all_cancers,
            "agents_to_call": agents,
        }

        print(f"\n🎯 Routing Plan:")
        print(f"   Cancer Type : {cancer_type or 'Not specified'}")
        print(f"   Intent      : {intent}")
        print(f"   Comparison  : {is_comparison}")
        print(f"   Agents      : {agents}")
        return plan

    def run(self, query: str) -> str:
        print(f"\n{'='*55}")
        print(f"🏥 MEDICAL RESEARCH ASSISTANT")
        print(f"{'='*55}")
        print(f"📝 Query: {query}")

        plan        = self.analyze_query(query)
        cancer_type = plan["cancer_type"]
        agents      = plan["agents_to_call"]

        diagnosis_output = treatment_output = summary_output = None

        if "diagnosis" in agents:
            diagnosis_output = self.diagnosis_agent.run(query, cancer_type)

        if "treatment" in agents:
            treatment_output = self.treatment_agent.run(query, cancer_type)

        if "summarization" in agents:
            summary_output = self.summarization_agent.run(query, cancer_type)

        if "summarization_comparison" in agents:
            summary_output = self.summarization_agent.run_comparison(
                query, plan["cancer_types_detected"]
            )

        # Merge if multiple agents ran
        if diagnosis_output and treatment_output:
            print(f"\n🔀 Merging diagnosis + treatment outputs...")
            final_response = self.summarization_agent.merge_agent_outputs(
                query, diagnosis_output, treatment_output
            )
        elif diagnosis_output:
            final_response = format_final_response(
                "diagnosis", cancer_type, diagnosis_output=diagnosis_output
            )
        elif treatment_output:
            final_response = format_final_response(
                "treatment", cancer_type, treatment_output=treatment_output
            )
        elif summary_output:
            final_response = format_final_response(
                "summarization", cancer_type, summary_output=summary_output
            )
        else:
            final_response = (
                "I was unable to retrieve relevant information. "
                "Please try rephrasing or check that PDFs are loaded."
            )

        print(f"\n{'='*55}")
        print(f"✅ Response generated")
        print(f"{'='*55}\n")

        return add_medical_disclaimer(final_response)

    def get_cancer_summary(self, cancer_type: str) -> str:
        return self.summarization_agent.summarize_cancer_type(cancer_type)
