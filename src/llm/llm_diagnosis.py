"""
LLM-Enhanced Malaria Diagnosis System
Uses GPT-4 with RAG for intelligent symptom interpretation
"""
import os
import pandas as pd
from typing import Dict, Tuple, List
from openai import OpenAI
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

class LLMMalariaExpert:
    """LLM-powered malaria diagnosis system with RAG"""
    
    def __init__(self, model="gpt-4-turbo-preview"):
        """Initialize with OpenAI API"""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.who_guidelines = self._load_who_guidelines()
    
    def _load_who_guidelines(self) -> str:
        """Load WHO malaria treatment guidelines"""
        # Simplified WHO guidelines (you can expand this)
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

1. SEVERITY_STAGE: Classify as one of:
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

5. CONFIDENCE: Your confidence level (0-100%)

Respond in JSON format:
{{
    "severity_stage": "...",
    "clinical_reasoning": "...",
    "diagnosis_text": "...",
    "prescription": "...",
    "confidence": ...
}}
"""
        return prompt
    
    def diagnose(self, symptoms: Dict[str, str], 
                 demographics: Dict[str, str]) -> Dict:
        """
        Diagnose malaria using LLM
        
        Args:
            symptoms: Dict of symptom_name -> 'yes'/'no'
            demographics: Patient demographics
        
        Returns:
            Dict with diagnosis results
        """
        # Create prompt
        prompt = self.create_diagnosis_prompt(symptoms, demographics)
        
        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert malaria diagnostician following WHO guidelines."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistent medical advice
                response_format={"type": "json_object"}
            )
            
            # Parse response
            result = json.loads(response.choices[0].message.content)
            
            # Add metadata
            result['model'] = self.model
            result['tokens_used'] = response.usage.total_tokens
            
            return result
            
        except Exception as e:
            print(f"❌ Error calling OpenAI API: {e}")
            # Fallback to conservative diagnosis
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
    Evaluate LLM system on dataset
    
    Args:
        dataset_path: Path to evaluation dataset
        sample_size: Optional - test on smaller sample first (to save API costs)
    """
    print("\n" + "="*70)
    print("🤖 LLM-ENHANCED MALARIA DIAGNOSIS EVALUATION")
    print("="*70 + "\n")
    
    # Load dataset
    df = pd.read_csv(dataset_path)
    
    if sample_size:
        print(f"⚠️  Using sample of {sample_size} cases (to manage API costs)")
        # Stratified sample
        df = df.groupby('severity_stage', group_keys=False).apply(
            lambda x: x.sample(min(len(x), sample_size // 4))
        ).reset_index(drop=True)
    
    print(f"📊 Evaluating on {len(df)} cases\n")
    
    # Initialize LLM system
    try:
        llm_system = LLMMalariaExpert()
        print("✅ LLM system initialized\n")
    except ValueError as e:
        print(f"❌ {e}")
        print("\n💡 Setup instructions:")
        print("   1. Get OpenAI API key from: https://platform.openai.com/api-keys")
        print("   2. Add to .env file: OPENAI_API_KEY=your-key-here")
        return
    
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
    total_cost = 0
    
    print("🔄 Processing cases...")
    for idx, row in df.iterrows():
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
        
        # Track cost (GPT-4 Turbo: ~$0.01 per 1K input tokens, $0.03 per 1K output)
        tokens = llm_result.get('tokens_used', 0)
        cost = (tokens / 1000) * 0.02  # Approximate
        total_cost += cost
        
        results.append({
            'case_id': row['case_id'],
            'expected': expected,
            'predicted': predicted,
            'correct': is_correct,
            'confidence': llm_result.get('confidence', 0),
            'reasoning': llm_result.get('clinical_reasoning', ''),
            'diagnosis': llm_result.get('diagnosis_text', ''),
            'prescription': llm_result.get('prescription', ''),
            'tokens': tokens,
            'cost': cost
        })
        
        # Progress indicator
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(df)} cases... (${total_cost:.2f} spent)")
    
    # Calculate metrics
    accuracy = (correct / len(df)) * 100
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Print results
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE")
    print("="*70)
    print(f"\n📈 Overall Accuracy: {accuracy:.2f}%")
    print(f"   Correct: {correct}/{len(df)}")
    print(f"   💰 Total API Cost: ${total_cost:.2f}")
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
    for stage in results_df['expected'].unique():
        if stage != 'All':
            stage_df = results_df[results_df['expected'] == stage]
            stage_acc = (stage_df['correct'].sum() / len(stage_df)) * 100
            print(f"   {stage:15s}: {stage_acc:5.1f}% ({stage_df['correct'].sum()}/{len(stage_df)})")
    
    # Save results
    output_path = 'data/processed/llm_results.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\n💾 Results saved to: {output_path}")
    
    # Show sample reasoning
    print(f"\n🔍 Sample LLM Reasoning (first 3 cases):")
    for idx, row in results_df.head(3).iterrows():
        print(f"\n  Case {row['case_id']}:")
        print(f"    Expected: {row['expected']} | Predicted: {row['predicted']} ✓" if row['correct'] else f"    Expected: {row['expected']} | Predicted: {row['predicted']} ✗")
        print(f"    Reasoning: {row['reasoning'][:150]}...")
    
    return results_df, accuracy

if __name__ == "__main__":
    import sys
    
    # Check for API key
    if not os.getenv('OPENAI_API_KEY'):
        print("\n⚠️  OpenAI API key not found!")
        print("Add to .env file: OPENAI_API_KEY=your-key-here\n")
        sys.exit(1)
    
    # Use combined dataset
    dataset_path = 'data/processed/malaria_combined_dataset.csv'
    
    # Start with small sample to test (20 cases)
    print("💡 Starting with SMALL SAMPLE (20 cases) to test setup")
    print("   This will cost approximately $0.40")
    print("   Edit sample_size parameter to test on full dataset\n")
    
    results, accuracy = evaluate_llm_system(dataset_path, sample_size=20)
    print("\n🎉 Evaluation finished!")