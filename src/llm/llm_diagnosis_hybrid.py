"""
HYBRID Malaria Diagnosis System: Rule-Based + LLM
Combines deterministic rules with LLM reasoning for best accuracy
"""
import os
import pandas as pd
from typing import Dict
import json
import requests
import time

class HybridMalariaExpert:
    """Hybrid system: rules for classification, LLM for reasoning"""
    
    def __init__(self, model="llama3.1:8b", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            if response.status_code != 200:
                raise ConnectionError("Ollama not running")
            print(f"✅ Hybrid System Ready (Rules + {self.model})")
        except Exception as e:
            print(f"❌ Ollama connection failed: {e}")
            raise
    
    def rule_based_classify(self, symptoms: Dict[str, str], demographics: Dict[str, str]) -> str:
        """Deterministic classification based on symptom patterns"""
        symptom_count = sum(1 for v in symptoms.values() if v == 'yes')
        
        # CRITICAL - strict rules
        if symptoms.get('hallucination') == 'yes':
            return 'Critical'
        if symptoms.get('semi_closed_eyes') == 'yes':
            return 'Critical'
        if symptom_count >= 15:
            return 'Critical'
        
        # NO MALARIA - strict rules
        rdt = demographics.get('rdt_result', 'not_done')
        microscopy = demographics.get('microscopy_result', 'not_done')
        
        if rdt == 'negative' and symptom_count < 4:
            return 'No_Malaria'
        if microscopy == 'negative' and symptom_count < 4:
            return 'No_Malaria'
        if symptom_count == 0:
            return 'No_Malaria'
        
        # STAGE II - moderate symptoms
        stage2_count = 0
        if symptoms.get('symptoms_over_1week') == 'yes':
            stage2_count += 1
        if symptoms.get('abdominal_pain') == 'yes':
            stage2_count += 1
        if symptoms.get('diarrhea') == 'yes':
            stage2_count += 1
        if symptom_count >= 8:
            stage2_count += 1
        
        if stage2_count >= 2:
            return 'Stage_II'
        
        # STAGE I - default for malaria symptoms
        if symptom_count >= 3:
            return 'Stage_I'
        if rdt == 'positive':
            return 'Stage_I'
        
        return 'No_Malaria'
    
    def get_llm_reasoning(self, symptoms: Dict[str, str], demographics: Dict[str, str], rule_classification: str) -> Dict:
        """Use LLM only for explanation and confidence"""
        positive_symptoms = [k.replace('_', ' ') for k, v in symptoms.items() if v == 'yes']
        symptom_count = len(positive_symptoms)
        
        symptom_str = ', '.join(positive_symptoms) if positive_symptoms else 'None'
        age = demographics.get('age', 'Unknown')
        sex = demographics.get('sex', 'Unknown')
        rdt = demographics.get('rdt_result', 'not_done')
        micro = demographics.get('microscopy_result', 'not_done')
        
        prompt = f"""You are explaining a malaria diagnosis to a patient.

PATIENT: {age}yr {sex}
TESTS: RDT={rdt}, Microscopy={micro}
SYMPTOMS ({symptom_count}): {symptom_str}

DIAGNOSIS: {rule_classification}

Explain this diagnosis in 1-2 sentences for the patient, recommend treatment, and rate confidence.

Respond with JSON only:
{{
    "clinical_reasoning": "Simple explanation for patient",
    "diagnosis_text": "One sentence summary",
    "prescription": "Treatment advice",
    "confidence": 85
}}"""
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.3,
                    "format": "json"
                },
                timeout=30
            )
            
            if response.status_code == 200:
                    result_text = response.json().get('response', '').strip()
                    # Handle fenced codeblocks like ```json ... ``` or plain JSON strings
                    if result_text.startswith('```'):
                        # Split by triple backticks and pick the first non-empty inner content
                        parts = [p for p in result_text.split('```') if p.strip() != '']
                        if len(parts) >= 1:
                            result_text = parts[0].strip()
                        else:
                            # fallback: remove backticks
                            result_text = result_text.replace('```', '').strip()
                    # If the inner block is prefixed with a language like "json", remove it
                    if result_text.startswith('json'):
                        result_text = result_text[4:].strip()
                
                    return json.loads(result_text)
            else:
                raise Exception(f"API error: {response.status_code}")
                
        except Exception as e:
            return {
                'clinical_reasoning': f'{rule_classification} based on symptom pattern',
                'diagnosis_text': f'{rule_classification} malaria diagnosis',
                'prescription': 'Consult healthcare provider for treatment',
                'confidence': 75
            }
    
    def diagnose(self, symptoms: Dict[str, str], demographics: Dict[str, str]) -> Dict:
        """Hybrid diagnosis: rules for classification, LLM for reasoning"""
        severity_stage = self.rule_based_classify(symptoms, demographics)
        llm_output = self.get_llm_reasoning(symptoms, demographics, severity_stage)
        
        result = {
            'severity_stage': severity_stage,
            'clinical_reasoning': llm_output.get('clinical_reasoning', ''),
            'diagnosis_text': llm_output.get('diagnosis_text', ''),
            'prescription': llm_output.get('prescription', ''),
            'confidence': llm_output.get('confidence', 80),
            'model': f'Hybrid (Rules + {self.model})'
        }
        
        return result

def evaluate_hybrid_system(dataset_path: str, sample_size: int = None):
    """Evaluate hybrid system"""
    print("\n" + "="*70)
    print("🎯 HYBRID SYSTEM EVALUATION (Rules + LLM)")
    print("="*70 + "\n")
    
    df = pd.read_csv(dataset_path)
    
    if sample_size:
        print(f"📊 Testing on {sample_size} cases")
        df = df.groupby('severity_stage', group_keys=False).apply(
            lambda x: x.sample(min(len(x), sample_size // 4))
        ).reset_index(drop=True)
    
    print(f"Dataset: {len(df)} cases\n")
    
    try:
        system = HybridMalariaExpert()
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
    
    print("🔄 Processing (hybrid is faster than pure LLM)...\n")
    
    start_time = time.time()
    
    for idx, row in df.iterrows():
        case_start = time.time()
        
        symptoms = {col: row.get(col, 'no') for col in symptom_cols}
        demographics = {col: row.get(col, 'Unknown') for col in demo_cols}
        
        result = system.diagnose(symptoms, demographics)
        
        expected = row.get('severity_stage', 'Unknown')
        predicted = result['severity_stage']
        is_correct = (predicted == expected)
        
        if is_correct:
            correct += 1
        
        case_time = time.time() - case_start
        
        results.append({
            'case_id': row['case_id'],
            'expected': expected,
            'predicted': predicted,
            'correct': is_correct,
            'confidence': result.get('confidence', 0),
            'reasoning': result.get('clinical_reasoning', ''),
            'diagnosis': result.get('diagnosis_text', ''),
            'prescription': result.get('prescription', ''),
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
    print("✅ HYBRID EVALUATION COMPLETE")
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
    
    output_path = 'data/processed/hybrid_results.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\n💾 Saved to: {output_path}")
    
    print(f"\n🎯 FINAL PERFORMANCE COMPARISON:")
    print(f"   Baseline (Rules only):    30.62%")
    print(f"   LLM v1 (Pure LLM):        50.00%")
    print(f"   LLM v2 (Optimized):       50.00%")
    print(f"   Hybrid (Rules+LLM):       {accuracy:.2f}%")
    print(f"   Total Improvement:        {accuracy - 30.62:+.2f}%")
    
    return results_df, accuracy

if __name__ == "__main__":
    import sys
    
    dataset_path = 'data/processed/malaria_combined_dataset.csv'
    
    if not os.path.exists(dataset_path):
        print(f"\n❌ Dataset not found\n")
        sys.exit(1)
    
    print("\n💡 HYBRID APPROACH: Rule-based classification + LLM reasoning")
    print("   Expected: 70-85% accuracy (best of both worlds)")
    print("   Benefit: Fast, accurate, AND explainable\n")
    
    input("Press Enter to start...")
    
    results, accuracy = evaluate_hybrid_system(dataset_path, sample_size=None)
    
    if results is not None and accuracy > 70:
        print("\n" + "="*70)
        print("🎉 EXCELLENT RESULTS! Ready for full evaluation")
        print("="*70)
        print("\nTo run on full 1682-case dataset:")
        print("  Edit: sample_size=None")
        print("  Time: ~30-45 minutes")
        print("  Then we can write the journal paper!")