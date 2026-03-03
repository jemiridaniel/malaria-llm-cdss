"""
Multi-LLM explanation service.
Fallback chain: Groq (llama-3.1-8b-instant) → Anthropic Claude → OpenAI → rule-based fallback
"""
import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class LLMResult:
    clinical_reasoning: str
    confidence: int
    model_used: str


def _build_prompt(
    demographics: Dict[str, str],
    severity_stage: str,
    positive_symptoms: List[str],
    diagnosis_text: str,
) -> str:
    symptom_str = ", ".join(positive_symptoms) if positive_symptoms else "none reported"
    return f"""You are a clinical decision support system explaining a malaria diagnosis to a patient.

PATIENT: Age {demographics.get("age", "unknown")}, {demographics.get("sex", "unknown")}
TEST RESULTS: RDT={demographics.get("rdt_result", "not done")}, Microscopy={demographics.get("microscopy_result", "not done")}
REPORTED SYMPTOMS ({len(positive_symptoms)}): {symptom_str}
DIAGNOSIS: {severity_stage} — {diagnosis_text}

Write a 2-sentence clinical explanation in plain language. Reference the specific symptoms the patient reported.

Respond with JSON only:
{{"clinical_reasoning": "Your explanation here referencing their specific symptoms", "confidence": 85}}"""


class LLMService:
    """Groq → Anthropic → OpenAI fallback chain for LLM explanations."""

    def explain(
        self,
        symptoms: Dict[str, str],
        demographics: Dict[str, str],
        severity_stage: str,
        positive_symptoms: List[str],
        diagnosis_text: str,
    ) -> LLMResult:
        prompt = _build_prompt(demographics, severity_stage, positive_symptoms, diagnosis_text)

        if settings.groq_api_key:
            result = self._try_groq(prompt)
            if result:
                return result

        if settings.anthropic_api_key:
            result = self._try_anthropic(prompt)
            if result:
                return result

        if settings.openai_api_key:
            result = self._try_openai(prompt)
            if result:
                return result

        return self._rule_based_fallback(severity_stage, positive_symptoms)

    # ── JSON parsing ──────────────────────────────────────────────────────────

    def _parse_json(self, text: str) -> Optional[Dict]:
        text = text.strip()
        if "```" in text:
            parts = [p for p in text.split("```") if p.strip()]
            if parts:
                text = parts[0].strip()
                if text.startswith("json"):
                    text = text[4:].strip()
        try:
            return json.loads(text)
        except Exception:
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end != -1:
                try:
                    return json.loads(text[start : end + 1])
                except Exception:
                    pass
        return None

    # ── Provider attempts ─────────────────────────────────────────────────────

    def _try_groq(self, prompt: str) -> Optional[LLMResult]:
        try:
            from groq import Groq

            client = Groq(api_key=settings.groq_api_key)
            response = client.chat.completions.create(
                model=settings.groq_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3,
            )
            data = self._parse_json(response.choices[0].message.content)
            if data:
                return LLMResult(
                    clinical_reasoning=data.get("clinical_reasoning", ""),
                    confidence=int(data.get("confidence", 80)),
                    model_used=f"groq/{settings.groq_model}",
                )
        except Exception as e:
            print(f"GROQ ERROR: {e}", flush=True)
        return None

    def _try_anthropic(self, prompt: str) -> Optional[LLMResult]:
        try:
            from anthropic import Anthropic

            client = Anthropic(api_key=settings.anthropic_api_key)
            response = client.messages.create(
                model=settings.anthropic_model,
                max_tokens=300,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )
            data = self._parse_json(response.content[0].text)
            if data:
                return LLMResult(
                    clinical_reasoning=data.get("clinical_reasoning", ""),
                    confidence=int(data.get("confidence", 80)),
                    model_used=f"anthropic/{settings.anthropic_model}",
                )
        except Exception as e:
            logger.warning(f"Anthropic failed: {e}")
        return None

    def _try_openai(self, prompt: str) -> Optional[LLMResult]:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=settings.openai_api_key)
            response = client.chat.completions.create(
                model=settings.openai_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            data = self._parse_json(response.choices[0].message.content)
            if data:
                return LLMResult(
                    clinical_reasoning=data.get("clinical_reasoning", ""),
                    confidence=int(data.get("confidence", 80)),
                    model_used=f"openai/{settings.openai_model}",
                )
        except Exception as e:
            logger.warning(f"OpenAI failed: {e}")
        return None

    # ── Rule-based fallback (no LLM required) ─────────────────────────────────

    def _rule_based_fallback(self, severity_stage: str, positive_symptoms: List[str]) -> LLMResult:
        symptom_str = ", ".join(positive_symptoms[:5]) if positive_symptoms else "no specific symptoms"
        messages = {
            "Critical": (
                f"Critical malaria has been detected based on severe symptoms including {symptom_str}. "
                "Immediate hospitalization is required — do not delay seeking emergency care."
            ),
            "Stage_II": (
                f"Moderate malaria progression is indicated by your symptoms of {symptom_str}. "
                "Seek immediate medical attention and follow your doctor's treatment plan."
            ),
            "Stage_I": (
                f"Early-stage malaria is suggested by your symptoms of {symptom_str}. "
                "Rest, stay well-hydrated, and take prescribed antimalarial medication."
            ),
            "No_Malaria": (
                f"Your current symptom pattern ({symptom_str}) does not match malaria criteria. "
                "Continue monitoring your condition and consult a doctor if symptoms worsen."
            ),
        }
        return LLMResult(
            clinical_reasoning=messages.get(severity_stage, "Please consult a healthcare provider."),
            confidence=70,
            model_used="rule-based-fallback",
        )
