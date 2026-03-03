from pydantic import BaseModel
from typing import List, Optional


class SymptomsInput(BaseModel):
    headache: str = "no"
    cough: str = "no"
    body_pain: str = "no"
    poor_appetite: str = "no"
    rash: str = "no"
    sweating: str = "no"
    chest_pain: str = "no"
    symptoms_less_1week: str = "no"
    abdominal_pain: str = "no"
    constipation: str = "no"
    restlessness: str = "no"
    diarrhea: str = "no"
    dizziness: str = "no"
    semi_closed_eyes: str = "no"
    exhaustion: str = "no"
    symptoms_over_1week: str = "no"
    hallucination: str = "no"
    back_pain: str = "no"
    blurry_vision: str = "no"


class DemographicsInput(BaseModel):
    age: str = "Unknown"
    sex: str = "Unknown"
    pregnant: str = "no"
    genotype: str = "AA"
    blood_type: str = "O+"
    rdt_result: str = "not_done"
    microscopy_result: str = "not_done"


class DiagnosisRequest(BaseModel):
    patient_name: str = ""
    patient_id: str = ""
    symptoms: SymptomsInput
    demographics: DemographicsInput


class DiagnosisResponse(BaseModel):
    severity_stage: str
    diagnosis_text: str
    clinical_reasoning: str
    prescription: str
    confidence: int
    positive_symptoms: List[str]
    symptom_count: int
    model_used: str
    timestamp: str
    patient_name: str = ""
    patient_id: str = ""


class ReportRequest(BaseModel):
    patient_name: str = ""
    patient_id: str = ""
    timestamp: str = ""
    demographics: DemographicsInput
    positive_symptoms: List[str]
    severity_stage: str
    diagnosis_text: str
    confidence: int
    clinical_reasoning: str
    prescription: str
    model_used: str
