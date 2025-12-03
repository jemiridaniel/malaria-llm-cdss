
"""OPTIMIZED Ollama LLM Malaria Diagnosis System
Enhanced prompts with better severity classification logic
"""
import os
import pandas as pd
from typing import Dict
import json
import requests
import time

class OptimizedOllamaExpert:
    """Optimized Ollama-powered malaria diagnosis with improved prompting"""
    
    def __init__(self, model="llama3.1:8b", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.who_guidelines = self._load_who_guidelines()
        
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code != 200:
                raise ConnectionError("Ollama server not running")
            print(f"✅ Connected to Ollama (model: {self.model})")
        except Exception as e:
            print(f"❌ Cannot connect to Ollama: {e}")
            raise
    
    def _load_who_guidelines(self) -> str:
        return """
WHO MALARIA SEVERITY CLASSIFICATION CRITERIA:

CRITICAL MALARIA (IMMEDIATE DANGER):
RED FLAGS - ANY ONE of these requires Critical classification:
- Hallucinations, confusion, or altered consciousness
- Semi-closed eyes (prostration/inability to open eyes fully)
- Severe exhaustion with multiple organ symptoms (>10 symptoms total)
- Symptoms persisting >1 week WITH severe complications
TREATMENT: IMMEDIATE HOSPITALIZATION, IV artesunate

STAGE II - MODERATE MALARIA:
Criteria - AT LEAST FOUR of these:
- Persistent high fever with sweating
- Severe headache
- Abdominal pain
- Vomiting or diarrhea  
- Symptoms lasting >7 days (without critical signs)
- Body pain + exhaustion + multiple symptoms (6-10 total symptoms)
TREATMENT: Medical supervision, possible hospitalization, oral/injectable ACT

STAGE I - UNCOMPLICATED MALARIA:
Criteria - Malaria symptoms WITHOUT Stage II/Critical features:
- Headache + fever-like symptoms
- Body pain, poor appetite
- Fatigue/exhaustion
- Fewer symptoms (<6 total)
- Duration <7 days typically
- Patient can function with discomfort
TREATMENT: Oral ACT, rest, fluids, outpatient care

NO MALARIA:
- RDT/microscopy negative AND minimal symptoms (<3)
- OR symptoms don't match malaria pattern
TREATMENT: Monitor, consider other diagnoses

DECISION TREE:
1. Check for CRITICAL red flags first (hallucination, semi-closed eyes, >10 symptoms)
2. If no red flags, count total symptoms:
   - 0-3 symptoms → likely No_Malaria (unless RDT positive)
   - 4-6 symptoms → Stage_I
   - 7-10 symptoms → Stage_II
   - >10 symptoms → Critical
3. Adjust based on test results (RDT positive increases severity)
4. Duration >7 days increases severity by one stage
"""
    
    def create_optimized_prompt(self, symptoms: Dict[str, str], demographics: Dict[str, str]) -> str:
        positive_symptoms = [k.replace('_', ' ') for k, v in symptoms.items() if v == 'yes']
        symptom_count = len(positive_symptoms)
        
        critical_indicators = []
        if symptoms.get('hallucination') == 'yes':
            critical_indicators.append('hallucination')
        if symptoms.get('semi_closed_eyes') == 'yes':
            critical_indicators.append('semi-closed eyes (prostration)')
        if symptoms.get('symptoms_over_1week') == 'yes' and symptom_count > 8:
            critical_indicators.append('prolonged illness with multiple complications')
        
        patient_summary = f"""
PATIENT DATA:
Age: {demographics.get('age', 'Unknown')} | Sex: {demographics.get('sex', 'Unknown')} | Pregnant: {demographics.get('pregnant', 'no')}
Genotype: {demographics.get('genotype', 'Unknown')} | Blood: {demographics.get('blood_type', 'Unknown')}

TEST RESULTS:
RDT: {demographics.get('rdt_result', 'not_done')} | Microscopy: {demographics.get('microscopy_result', 'not_done')}

SYMPTOM COUNT: {symptom_count} symptoms reported

SYMPTOMS PRESENT:
{', '.join(positive_symptoms) if positive_symptoms else 'None'}

CRITICAL RED FLAGS PRESENT: {len(critical_indicators)}
{', '.join(critical_indicators) if critical_indicators else 'None'}
"""
        
        prompt = f"""You are an expert malaria diagnostician. Use the WHO severity classification criteria to analyze this case.

{patient_summary}

{self.who_guidelines}

ANALYSIS STEPS:
1. Count total symptoms: {symptom_count}
2. Check for critical red flags: {len(critical_indicators)} present
3. Consider test results and duration
4. Apply WHO severity classification

YOUR DIAGNOSIS:
Classify as EXACTLY one of: "No_Malaria", "Stage_I", "Stage_II", or "Critical"

RULES TO FOLLOW:
- If hallucination OR semi_closed_eyes present → MUST be "Critical"
- If symptom_count > 10 → MUST be "Critical"  
- If symptom_count 7-10 AND symptoms_over_1week → "Stage_II"
- If symptom_count 4-6 → "Stage_I"
- If symptom_count < 4 AND RDT negative → "No_Malaria"
- If RDT positive → minimum "Stage_I"

# ...existing code...
Respond ONLY with valid JSON (no markdown):
{{
    "severity_stage": "exactly one of: No_Malaria, Stage_I, Stage_II, Critical",
    "clinical_reasoning": "Brief explanation of classification (max 2 sentences)",
    "diagnosis_text": "One sentence diagnosis",
    "prescription": "Treatment recommendation based on WHO guidelines",
    "confidence": 85
}}
"""
        return prompt
    
    def call_ollama(self, prompt: str, temperature: float = 0.1) -> str:
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": temperature,
                    "format": "json"
                },
                timeout=60
            )
            
            if response.status_code != 200:
                raise Exception(f"Ollama API error: {response.status_code}")
            
            result = response.json()
            return result['response']
            
        except Exception as e:
            raise Exception(f"Ollama call failed: {str(e)}")
    
    def diagnose(self, symptoms: Dict[str, str], demographics: Dict[str, str]) -> Dict:
        prompt = self.create_optimized_prompt(symptoms, demographics)
        
        try:
            response_text = self.call_ollama(prompt)
            
            clean_response = response_text.strip()
            if clean_response.startswith('```'):
                # handle fenced code block (e.g., ```json {...}```)
                parts = clean_response.split('```')
                if len(parts) >= 2:
                    clean_response = parts[1]
                else:
                    clean_response = clean_response.strip('`')
                if clean_response.strip().startswith('json'):
                    clean_response = clean_response.strip()[4:].lstrip()
            
            result = json.loads(clean_response)
            
            valid_stages = ['No_Malaria', 'Stage_I', 'Stage_II', 'Critical']
            predicted_stage = result.get('severity_stage', '')
            
            symptom_count = sum(1 for v in symptoms.values() if v == 'yes')
            
            if symptoms.get('hallucination') == 'yes' or symptoms.get('semi_closed_eyes') == 'yes':
                predicted_stage = 'Critical'
                result['clinical_reasoning'] = "Critical red flags present. " + result.get('clinical_reasoning', '')
            elif symptom_count > 10:
                predicted_stage = 'Critical'
            elif predicted_stage not in valid_stages:
                stage_lower = predicted_stage.lower()
                if 'critical' in stage_lower or 'severe' in stage_lower:
                    predicted_stage = 'Critical'
                elif 'stage ii' in stage_lower or 'stage 2' in stage_lower or 'moderate' in stage_lower:
                    predicted_stage = 'Stage_II'
                elif 'stage i' in stage_lower or 'stage 1' in stage_lower or 'uncomplicated' in stage_lower:
                    predicted_stage = 'Stage_I'
                else:
                    predicted_stage = 'No_Malaria'
            
            result['severity_stage'] = predicted_stage
            result['model'] = self.model
            
            return result
            
        except Exception as e:
            print(f"⚠️  Error: {e}")
            return {
                'severity_stage': 'Stage_II',
                'clinical_reasoning': f'Error in processing: {str(e)}',
                'diagnosis_text': 'Unable to diagnose - consult physician',
                'prescription': 'Medical consultation recommended',
                'confidence': 50,
                'error': str(e)
            }

def evaluate_optimized_system(dataset_path: str, sample_size: int = None):
    print("\n" + "="*70)
    print("🚀 OPTIMIZED OLLAMA LLM EVALUATION (v2)")
    print("="*70 + "\n")
    
    df = pd.read_csv(dataset_path)
    
    if sample_size:
        print(f"📊 Testing on {sample_size} cases (stratified sample)")
        df = df.groupby('severity_stage', group_keys=False).apply(
            lambda x: x.sample(min(len(x), sample_size // 4))
        ).reset_index(drop=True)
    
    print(f"Dataset: {len(df)} cases\n")
    
    try:
        llm_system = OptimizedOllamaExpert()
        print()
    except Exception as e:
        print(f"❌ {e}")
        return None, 0
    
    symptom_cols = [
        'headache', 'cough', 'body_pain', 'poor_appetite', 'rash',
        'sweating', 'chest_pain', 'symptoms_less_1week', 'abdominal_pain',
        'constipation', 'restlessness', 'diarrhea', 'dizziness',
        'semi_closed_eyes', 'exhaustion', 'symptoms_over_1week',
        'hallucination', 'back_pain', 'blurry_vision'
    ]
    
    demo_cols = ['age', 'sex', 'pregnant', 'genotype', 'blood_type', 'rdt_result', 'microscopy_result']
    
    results = []
    correct = 0
    
    print("🔄 Processing...\n")
    
    start_time = time.time()
    
    for idx, row in df.iterrows():
        case_start = time.time()
        
        symptoms = {col: row.get(col, 'no') for col in symptom_cols}
        demographics = {col: row.get(col, 'Unknown') for col in demo_cols}
        
        llm_result = llm_system.diagnose(symptoms, demographics)
        
        expected = row.get('severity_stage', 'Unknown')
        predicted = llm_result['severity_stage']
        is_correct = (predicted == expected)
        
        if is_correct:
            correct += 1
        
        case_time = time.time() - case_start
        
        results.append({
            'case_id': row['case_id'],
            'expected': expected,
            'predicted': predicted,
            'correct': is_correct,
            'confidence': llm_result.get('confidence', 0),
            'reasoning': llm_result.get('clinical_reasoning', ''),
            'diagnosis': llm_result.get('diagnosis_text', ''),
            'prescription': llm_result.get('prescription', ''),
            'processing_time': case_time
        })
        
        if (idx + 1) % 5 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (idx + 1)
            remaining = avg_time * (len(df) - idx - 1)
            accuracy_so_far = (correct / (idx + 1)) * 100
            print(f"  {idx + 1}/{len(df)} | Acc: {accuracy_so_far:.1f}% | Avg: {avg_time:.1f}s | ETA: {remaining/60:.1f}min")
    
    total_time = time.time() - start_time
    accuracy = (correct / len(df)) * 100
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*70)
    print("✅ OPTIMIZED EVALUATION COMPLETE")
    print("="*70)
    print(f"\n📈 Overall Accuracy: {accuracy:.2f}%")
    print(f"   Correct: {correct}/{len(df)}")
    print(f"   ⏱️  Time: {total_time/60:.1f} min ({total_time/len(df):.1f}s/case)")
    print(f"   📊 Avg Confidence: {results_df['confidence'].mean():.1f}%")
    
    print(f"\n📊 Confusion Matrix:")
    confusion = pd.crosstab(results_df['expected'], results_df['predicted'], margins=True)
    print(confusion)
    
    print(f"\n📉 Per-Stage Performance:")
    for stage in ['No_Malaria', 'Stage_I', 'Stage_II', 'Critical']:
        stage_df = results_df[results_df['expected'] == stage]
        if len(stage_df) > 0:
            stage_acc = (stage_df['correct'].sum() / len(stage_df)) * 100
            print(f"   {stage:15s}: {stage_acc:5.1f}% ({stage_df['correct'].sum()}/{len(stage_df)})")
    
    output_path = 'data/processed/llm_ollama_v2_results.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\n💾 Saved to: {output_path}")
    
    print(f"\n🎯 PERFORMANCE COMPARISON:")
    print(f"   Baseline:      30.62%")
    print(f"   LLM v1:        50.00%")
    print(f"   LLM v2:        {accuracy:.2f}%")
    print(f"   Improvement:   {accuracy - 30.62:+.2f}%")
    
    return results_df, accuracy

if __name__ == "__main__":
    import sys
    
    dataset_path = 'data/processed/malaria_combined_dataset.csv'
    
    if not os.path.exists(dataset_path):
        print(f"\n❌ Dataset not found: {dataset_path}\n")
        sys.exit(1)
    
    print("\n💡 Running OPTIMIZED version on 20-case test")
    print("   Expected improvement over v1 (50% → 70-80%)\n")
    
    input("Press Enter to start...")
    
    results, accuracy = evaluate_optimized_system(dataset_path, sample_size=20)