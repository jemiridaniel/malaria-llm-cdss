import logging
from datetime import datetime
from io import BytesIO

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.models.schemas import DiagnosisRequest, DiagnosisResponse, ReportRequest
from app.services.diagnosis_service import (
    get_positive_symptoms,
    get_stage_info,
    rule_based_classify,
)
from app.services.llm_service import LLMService
from app.services.report import generate_report

router = APIRouter(tags=["diagnosis"])
llm_service = LLMService()
logger = logging.getLogger(__name__)


@router.post("/diagnose", response_model=DiagnosisResponse)
async def diagnose(request: DiagnosisRequest):
    try:
        symptoms = request.symptoms.model_dump()
        demographics = request.demographics.model_dump()

        # Deterministic classification (rule-based engine)
        severity_stage = rule_based_classify(symptoms, demographics)
        diagnosis_text, prescription = get_stage_info(severity_stage)
        positive_symptoms = get_positive_symptoms(symptoms)

        # LLM explanation with fallback chain
        llm_result = llm_service.explain(
            symptoms, demographics, severity_stage, positive_symptoms, diagnosis_text
        )

        return DiagnosisResponse(
            severity_stage=severity_stage,
            diagnosis_text=diagnosis_text,
            clinical_reasoning=llm_result.clinical_reasoning,
            prescription=prescription,
            confidence=llm_result.confidence,
            positive_symptoms=positive_symptoms,
            symptom_count=len(positive_symptoms),
            model_used=llm_result.model_used,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            patient_name=request.patient_name,
            patient_id=request.patient_id,
        )
    except Exception as e:
        logger.error(f"Diagnosis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/report")
async def report(request: ReportRequest):
    try:
        pdf_bytes = generate_report(request.model_dump())
        patient_slug = request.patient_id or request.patient_name or "patient"
        # sanitise filename
        safe_slug = "".join(c if c.isalnum() or c in "-_" else "-" for c in patient_slug)
        filename = f"malaria-report-{safe_slug}.pdf"
        return StreamingResponse(
            BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as e:
        logger.error(f"Report generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
