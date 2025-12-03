"""
LLM-Enhanced Malaria Diagnosis System using Ollama (Free, Local)
Uses Llama 3.1 with RAG for intelligent symptom interpretation
"""
import os
import pandas as pd
from typing import Dict, List
import json
import requests

class OllamaLLMExpert:
    """Ollama-powered malaria diagnosis system with local LLM"""
    
    def __init__(self, model="llama3.1:8b", base_url="http://localhost:11434"):
        """Initialize with Ollama"""
        self.model = model
        self.base_url = base_url
        self.who_guidelines = self._load_who_guidelines()
        
        # Test connection
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code != 200:
                raise ConnectionError("Ollama server not running")
            print(f"✅ Connected to Ollama (model: {self.model})")
        except Exception as e:
            print(f"❌ Cannot connect to Ollama: {e}")
            print("💡 Start Ollama with: ollama serve")
            raise

    def _map_severity_stage(self, stage_input: str) -> str:
        """
        Maps raw severity stage from LLM to a standardized valid stage.
        Handles common variations and provides a fallback.
        """
        valid_stages = {'No_Malaria', 'Stage_I', 'Stage_II', 'Critical'}
        
        if stage_input in valid_stages:
            return stage_input
        
        # Try to map common variations (case-insensitive)
        stage = stage_input.lower()
        if 'critical' in stage or 'severe' in stage:
            return 'Critical'
        elif 'stage ii' in stage or 'stage 2' in stage or 'moderate' in stage:
            return 'Stage_II'
        elif 'stage i' in stage or 'stage 1' in stage or 'uncomplicated' in stage:
            return 'Stage_I'
        else:
            print(f"⚠️  Warning: Unrecognized severity stage '{stage_input}' from LLM. Defaulting to 'No_Malaria'.")
            return 'No_Malaria'
    
    def _load_who_guidelines(self) -> str:
        """Load WHO malaria treatment guidelines"""
        return """
WHO MALARIA DIAGNOSIS AND TREATMENT GUIDELINES (Simplified):

SEVERITY CLASSIFICATION:

1. UNCOMPLICATED MALARIA (Stage I):
- Symptoms: Fever, headache, body aches, fatigue, mild nausea
- Duration: < 7 days typically
- No severe complications
- Treatment: Oral artemisinin-based combination therapy (ACT)
- Prescription: Rest, oral fluids, antipyretics, ACT medication

2. MODERATE MALARIA (Stage II):
- Symptoms: Persistent high fever (>39°C), severe headache, vomiting,
  abdominal pain, weakness, multiple symptoms
- Duration: > 7 days or worsening symptoms
- Some organ dysfunction but no life-threatening complications
- Treatment: Close monitoring, IV fluids if needed, ACT or injectable artemisinin
- Prescription: Medical supervision, possible hospitalization

3. SEVERE/CRITICAL MALARIA:
- Symptoms: Altered consciousness, prostration, multiple convulsions,
  respiratory distress, circulatory collapse, jaundice, severe anemia,
  renal impairment, hypoglycemia, metabolic acidosis
- Danger signs: Inability to sit/stand, confusion, semi-consciousness,
  hallucinations, difficulty breathing
- Treatment: IMMEDIATE HOSPITALIZATION, IV artesunate, intensive care
- Prescription: Emergency care, intensive monitoring

SPECIAL CONSIDERATIONS:
- Pregnancy: Use quinine in first trimester, ACT in 2nd/3rd trimester
- Children < 5 years: Higher risk, lower threshold for hospitalization
- Elderly: Monitor for complications
- Sickle cell (SS genotype): Higher risk of severe malaria

DIAGNOSTIC CRITERIA:
- Clinical diagnosis based on symptom pattern
- Confirmed by: RDT (Rapid Diagnostic Test) or microscopy
- Consider malaria in any febrile patient in endemic areas
"""
    
    def create_diagnosis_prompt(self, symptoms: Dict[str, str], 
                               demographics: Dict[str, str]) -> str:
        """Create structured prompt for LLM diagnosis"""
        
        # Extract positive symptoms
        positive_symptoms = [k.replace('_', ' ') for k, v in symptoms.items() if v == 'yes']
        
        # Build patient profile
        patient_profile = f"""
PATIENT PROFILE:
- Age: {demographics.get('age', 'Unknown')} years
- Sex: {demographics.get('sex', 'Unknown')}
- Pregnant: {demographics.get('pregnant', 'no')}
- Genotype: {demographics.get('genotype', 'Unknown')}
- Blood Type: {demographics.get('blood_type', 'Unknown')}

DIAGNOSTIC TEST RESULTS:
- RDT Result: {demographics.get('rdt_result', 'not_done')}
- Microscopy: {demographics.get('microscopy_result', 'not_done')}

REPORTED SYMPTOMS ({len(positive_symptoms)} total):
{', '.join(positive_symptoms) if positive_symptoms else 'None reported'}
"""
        
        prompt = f"""You are an expert malaria diagnostician with access to WHO treatment guidelines. 
Analyze the following patient case and provide a comprehensive diagnosis.

{patient_profile}

REFERENCE GUIDELINES:
{self.who_guidelines}

TASK:
Based on the patient's symptoms, demographics, and test results, provide:

1. SEVERITY_STAGE: Classify as EXACTLY one of these four options:
   - "No_Malaria" (if symptoms don't match malaria or tests negative)
   - "Stage_I" (uncomplicated malaria, early stage)
   - "Stage_II" (moderate malaria, requires medical attention)
   - "Critical" (severe malaria, life-threatening)

2. CLINICAL_REASONING: Explain your diagnostic reasoning (2-3 sentences):
   - Which symptoms are most indicative?
   - Why this severity level?
   - Any risk factors?

3. DIAGNOSIS_TEXT: One-sentence clinical diagnosis

4. PRESCRIPTION: Detailed treatment recommendation based on WHO guidelines

5. CONFIDENCE: Your confidence level (0-100 as a number)

IMPORTANT: Respond ONLY with valid JSON in this exact format (no markdown, no code blocks):
{{
    "severity_stage": "Stage_I",
    "clinical_reasoning": "Patient presents with...",
    "diagnosis_text": "Early stage malaria...",
    "prescription": "Treatment recommendation...",
    "confidence": 85
}}
"""
        return prompt
    
    def call_ollama(self, prompt: str, temperature: float = 0.1) -> str:
        """Call Ollama API"""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": temperature,
                    "format": "json"  # Request JSON format
                },
                timeout=60  # 60 second timeout
            )
            
            if response.status_code != 200:
                raise Exception(f"Ollama API error: {response.status_code}")
            
            result = response.json()
            return result['response']
            
        except requests.exceptions.Timeout:
            raise Exception("Ollama request timeout - model may be too slow")
        except Exception as e:
            raise Exception(f"Ollama API call failed: {str(e)}")
    
    def diagnose(self, symptoms: Dict[str, str], 
                 demographics: Dict[str, str]) -> Dict:
        """
        Diagnose malaria using Ollama LLM
        
        Args:
            symptoms: Dict of symptom_name -> 'yes'/'no'
            demographics: Patient demographics
        
        Returns:
            Dict with diagnosis results
        """
        # Create prompt
        prompt = self.create_diagnosis_prompt(symptoms, demographics)
        
        try:
            # Call Ollama
            response_text = self.call_ollama(prompt)
            
            # Parse JSON response
            # Clean up response (remove markdown code blocks if present)
            clean_response = response_text.strip()
if clean_response.startswith('```'):
                # Remove markdown code blocks
                clean_response = clean_response.split('```')[1]
                if clean_response.startswith('json'):
                    clean_response = clean_response[4:]
            
            result = json.loads(clean_response)
            
            # Validate and map severity_stage
result['severity_stage'] = self._map_severity_stage(result.get('severity_stage'))
            
            # Add metadata
result['model'] = self.model
            
        return result
            
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON parse error: {e}")
            print(f"Response was: {response_text[:200]}...")
            # Fallback diagnosis
            return {
                'severity_stage': 'Stage_II',
                'clinical_reasoning': 'Unable to parse LLM response - conservative diagnosis recommended',
                'diagnosis_text': 'Possible malaria - please consult physician for confirmation',
                'prescription': 'Immediate medical consultation recommended',
                'confidence': 50,
                'error': f'JSON parse error: {str(e)}'
            }
        except Exception as e:
            print(f"❌ Error in diagnosis: {e}")
            return {
                'severity_stage': 'Stage_II',
                'clinical_reasoning': f'Error in LLM processing: {str(e)}',
                'diagnosis_text': 'Unable to diagnose - please consult physician',
                'prescription': 'Immediate medical consultation recommended',
                'confidence': 0,
                'error': str(e)
            }

def evaluate_llm_system(dataset_path: str, sample_size: int = None):
    """
    Evaluate Ollama LLM system on dataset
    
    Args:
        dataset_path: Path to evaluation dataset
        sample_size: Optional - test on smaller sample first
    """
    print("\n" + "="*70)
    print("🤖 OLLAMA LLM-ENHANCED MALARIA DIAGNOSIS EVALUATION")
    print("="*70 + "\n")
    
    # Load dataset
    df = pd.read_csv(dataset_path)
    
    if sample_size:
        print(f"⚠️  Using sample of {sample_size} cases for testing")
        # Stratified sample
        df = df.groupby('severity_stage', group_keys=False).apply(
            lambda x: x.sample(min(len(x), sample_size // 4))
        ).reset_index(drop=True)
    
    print(f"📊 Evaluating on {len(df)} cases")
    print(f"💰 Cost: FREE (running locally with Ollama)\n")
    
    # Initialize LLM system
    try:
        llm_system = OllamaLLMExpert()
        print()
    except Exception as e:
        print(f"❌ Failed to initialize Ollama: {e}")
        return None, 0
    
    # Symptom columns
    symptom_cols = [
        'headache', 'cough', 'body_pain', 'poor_appetite', 'rash',
        'sweating', 'chest_pain', 'symptoms_less_1week', 'abdominal_pain',
        'constipation', 'restlessness', 'diarrhea', 'dizziness',
        'semi_closed_eyes', 'exhaustion', 'symptoms_over_1week',
        'hallucination', 'back_pain', 'blurry_vision'
    ]
    
    # Demographic columns
    demo_cols = ['age', 'sex', 'pregnant', 'genotype', 'blood_type', 
                 'rdt_result', 'microscopy_result']
    
    # Evaluate each case
    results = []
    correct = 0
    
    print("🔄 Processing cases (this may take a while with local LLM)...\n")
    
    import time
    start_time = time.time()
    
    for idx, row in df.iterrows():
        case_start = time.time()
        
        # Extract data
        symptoms = {col: row.get(col, 'no') for col in symptom_cols}
        demographics = {col: row.get(col, 'Unknown') for col in demo_cols}
        
        # Diagnose
        llm_result = llm_system.diagnose(symptoms, demographics)
        
        # Compare with expected
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
        
        # Progress indicator
        if (idx + 1) % 5 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (idx + 1)
            remaining = avg_time * (len(df) - idx - 1)
            accuracy_so_far = (correct / (idx + 1)) * 100
            print(f"  {idx + 1}/{len(df)} cases | Accuracy: {accuracy_so_far:.1f}% | "
                  f"Avg: {avg_time:.1f}s/case | ETA: {remaining/60:.1f}min")
    
    # Calculate metrics
    total_time = time.time() - start_time
    accuracy = (correct / len(df)) * 100
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Print results
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE")
    print("="*70)
    print(f"\n📈 Overall Accuracy: {accuracy:.2f}%")
    print(f"   Correct: {correct}/{len(df)}")
    print(f"   ⏱️  Total Time: {total_time/60:.1f} minutes")
    print(f"   ⚡ Avg Time: {total_time/len(df):.1f} seconds/case")
    print(f"   📊 Average Confidence: {results_df['confidence'].mean():.1f}%")
    
    print(f"\n📊 Confusion Matrix:")
    confusion = pd.crosstab(
        results_df['expected'],
        results_df['predicted'],
        margins=True
    )
    print(confusion)
    
    # Per-stage accuracy
    print(f"\n📉 Per-Stage Performance:")
    for stage in ['No_Malaria', 'Stage_I', 'Stage_II', 'Critical']:
        stage_df = results_df[results_df['expected'] == stage]
        if len(stage_df) > 0:
            stage_acc = (stage_df['correct'].sum() / len(stage_df)) * 100
            print(f"   {stage:15s}: {stage_acc:5.1f}% ({stage_df['correct'].sum()}/{len(stage_df)})")
    
    # Save results
    output_path = 'data/processed/llm_ollama_results.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\n💾 Results saved to: {output_path}")
    
    # Show sample reasoning
    print(f"\n🔍 Sample LLM Reasoning (first 3 cases):")
    for idx, row in results_df.head(3).iterrows():
        status = "✓" if row['correct'] else "✗"
        print(f"\n  Case {row['case_id']} {status}:")
        print(f"    Expected: {row['expected']} | Predicted: {row['predicted']}")
        print(f"    Confidence: {row['confidence']}%")
        print(f"    Reasoning: {row['reasoning'][:120]}...")
    
    return results_df, accuracy

if __name__ == "__main__":
    import sys
    
    # Check if Ollama is running
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code != 200:
            print("\n❌ Ollama is not running!")
            print("Start it with: ollama serve")
            print("Or in a new terminal: ollama run llama3.1:8b\n")
            sys.exit(1)
    except:
        print("\n❌ Cannot connect to Ollama!")
        print("Make sure Ollama is running:")
        print("  1. Open new terminal")
        print("  2. Run: ollama serve")
        print("  3. Then run this script again\n")
        sys.exit(1)
    
    # Use combined dataset
    dataset_path = 'data/processed/malaria_combined_dataset.csv'
    
    if not os.path.exists(dataset_path):
        print(f"\n❌ Dataset not found: {dataset_path}")
        print("Run: python src/data/download_kaggle_data.py\n")
        sys.exit(1)
    
    # Start with small sample to test (20 cases)
    print("\n💡 Starting with SMALL SAMPLE (20 cases) to test setup")
    print("   This is FREE but may take 5-10 minutes")
    print("   Edit sample_size=None to run on full dataset (1682 cases)\n")
    
    input("Press Enter to continue...")
    
    results, accuracy = evaluate_llm_system(dataset_path, sample_size=20)
    
    if results is not None:
        print("\n" + "="*70)
        print("🎯 NEXT STEPS:")
        print("="*70)
        print("\n1. Review results in: data/processed/llm_ollama_results.csv")
        print("\n2. To run full evaluation (1682 cases):")
        print("   Edit line: sample_size=None")
        print("   Estimated time: 45-90 minutes")
        print("\n3. Compare with baseline (30.62% accuracy)")
        print(f"   LLM Accuracy: {accuracy:.2f}%")
        print(f"   Improvement: {accuracy - 30.62:+.2f}%")
