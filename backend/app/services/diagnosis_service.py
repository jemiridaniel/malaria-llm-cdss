"""
Malaria diagnosis service.

Classification logic ported verbatim from:
  src/llm/llm_diagnosis_hybrid.py — HybridMalariaExpert.rule_based_classify()

No rules have been modified. The algorithm is identical to the version that
achieved 87.22% accuracy in the research evaluation.
"""
from typing import Dict, List, Tuple


def rule_based_classify(symptoms: Dict[str, str], demographics: Dict[str, str]) -> str:
    """Deterministic classification based on symptom patterns."""
    symptom_count = sum(1 for v in symptoms.values() if v == "yes")

    # CRITICAL — strict rules
    if symptoms.get("hallucination") == "yes":
        return "Critical"
    if symptoms.get("semi_closed_eyes") == "yes":
        return "Critical"
    if symptom_count >= 15:
        return "Critical"

    # NO MALARIA — strict rules
    rdt = demographics.get("rdt_result", "not_done")
    microscopy = demographics.get("microscopy_result", "not_done")

    if rdt == "negative" and symptom_count < 4:
        return "No_Malaria"
    if microscopy == "negative" and symptom_count < 4:
        return "No_Malaria"
    if symptom_count == 0:
        return "No_Malaria"

    # STAGE II — moderate symptoms
    stage2_count = 0
    if symptoms.get("symptoms_over_1week") == "yes":
        stage2_count += 1
    if symptoms.get("abdominal_pain") == "yes":
        stage2_count += 1
    if symptoms.get("diarrhea") == "yes":
        stage2_count += 1
    if symptom_count >= 8:
        stage2_count += 1

    if stage2_count >= 2:
        return "Stage_II"

    # STAGE I — default for malaria symptoms
    if symptom_count >= 3:
        return "Stage_I"
    if rdt == "positive":
        return "Stage_I"

    return "No_Malaria"


def get_positive_symptoms(symptoms: Dict[str, str]) -> List[str]:
    """Return human-readable list of reported positive symptoms."""
    label_map = {
        "headache": "headache",
        "cough": "cough",
        "body_pain": "body pain",
        "poor_appetite": "poor appetite",
        "rash": "rash",
        "sweating": "sweating",
        "chest_pain": "chest pain",
        "symptoms_less_1week": "symptoms < 1 week",
        "abdominal_pain": "abdominal pain",
        "constipation": "constipation",
        "restlessness": "restlessness",
        "diarrhea": "diarrhea",
        "dizziness": "dizziness",
        "semi_closed_eyes": "eyes stay semi-closed",
        "exhaustion": "exhaustion",
        "symptoms_over_1week": "symptoms > 1 week",
        "hallucination": "hallucination",
        "back_pain": "back pain",
        "blurry_vision": "blurry vision",
    }
    return [label_map.get(k, k.replace("_", " ")) for k, v in symptoms.items() if v == "yes"]


def get_stage_info(stage: str) -> Tuple[str, str]:
    """Return (diagnosis_text, prescription) for a given severity stage."""
    info = {
        "Stage_I": (
            "Stage I Malaria: Early stage malaria detected",
            "Drink a lot of fluid. Do not have ice in drinks. Rest adequately. "
            "Take prescribed antimalarial medication.",
        ),
        "Stage_II": (
            "Stage II Malaria: Moderate malaria requiring medical attention",
            "The fever typically levels off at a high temperature (between 39–40°C). "
            "Immediate medical consultation required.",
        ),
        "Critical": (
            "CRITICAL: Severe malaria complications detected",
            "Malaria has been determined to be severely critical. Please visit a doctor "
            "immediately. Self-medication is dangerous. IMMEDIATE HOSPITALIZATION REQUIRED.",
        ),
        "No_Malaria": (
            "No malaria symptoms detected",
            "Continue monitoring your symptoms. Consult a doctor if symptoms persist or worsen.",
        ),
    }
    return info.get(stage, ("Unknown stage", "Consult a healthcare provider."))
