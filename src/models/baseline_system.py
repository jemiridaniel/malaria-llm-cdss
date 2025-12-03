"""
Baseline malaria diagnosis system - Python version of original PHP logic
Rule-based expert system using forward chaining
"""
import pandas as pd
import os
from typing import Dict, List, Tuple

class MalariaExpertSystem:
    """Rule-based expert system for malaria diagnosis"""
    
    def __init__(self):
        self.questions = self._load_questions()
        self.rules = self._define_rules()
    
    def _load_questions(self) -> List[str]:
        """19 diagnostic questions from original system"""
        return [
            'Do you have headache?',
            'Is cough one of your symptoms?',
            'Are you experiencing body pain?',
            'Do you think you have poor appetite?',
            'Do you have rash?',
            'Is the symptoms accompanied with sweat?',
            'Are you experiencing chest Pain?',
            'Has the symptoms you are experiencing been less 1-week?',
            'Are you having abdominal pain?',
            'Are you having constipation?',
            'Are you restless?',
            'Are you having bowel movement(diarrhea)?',
            'Do you feel dizzy?',
            'Can you keep your eyes wide open or they stay semi-closed?',
            'Any feeling of exhaustion?',
            'Has the symptoms been over 1-week?',
            'Do you think you are hallucinating?',
            'Are you having backpain?',
            'Are you having blurry vision?'
        ]
    
    def _define_rules(self) -> Dict:
        """Define diagnostic rules based on original PHP system"""
        return {
            'stage_1': {
                'required': ['headache', 'body_pain', 'poor_appetite', 
                           'symptoms_less_1week', 'exhaustion'],
                'optional': ['cough', 'dizziness', 'sweating'],
                'min_required': 5,
                'diagnosis': 'Stage I Malaria: Early stage malaria detected',
                'prescription': 'Drink a lot of fluid. Do not have ice in drinks. Rest adequately. Take prescribed antimalarial medication.'
            },
            'stage_2': {
                'required': ['headache', 'cough', 'body_pain', 'sweating',
                           'abdominal_pain', 'exhaustion', 'symptoms_over_1week'],
                'optional': ['dizziness', 'restlessness', 'chest_pain'],
                'min_required': 7,
                'diagnosis': 'Stage II Malaria: Moderate malaria requiring medical attention',
                'prescription': 'The fever typically levels off at a high temperature (between 39 and 40 degrees C). Immediate medical consultation required.'
            },
            'critical': {
                'required': ['headache', 'semi_closed_eyes', 'hallucination',
                           'dizziness', 'exhaustion', 'symptoms_over_1week'],
                'optional': ['all'],
                'min_required': 6,
                'diagnosis': 'CRITICAL: Severe malaria complications detected',
                'prescription': 'Malaria level has been determined to severely be critical. Please visit a doctor as soon as possible. Self Medication can be dangerous to your health. IMMEDIATE HOSPITALIZATION REQUIRED.'
            }
        }
    
    def diagnose(self, symptoms: Dict[str, str]) -> Tuple[str, str, str]:
        """
        Diagnose malaria based on symptoms
        
        Args:
            symptoms: Dict mapping symptom names to 'yes'/'no'
        
        Returns:
            Tuple of (severity_stage, diagnosis, prescription)
        """
        # Count yes symptoms
        yes_symptoms = [k for k, v in symptoms.items() if v == 'yes']
        symptom_count = len(yes_symptoms)
        
        # Check critical first (highest priority)
        if self._matches_rule('critical', symptoms):
            rule = self.rules['critical']
            return ('Critical', rule['diagnosis'], rule['prescription'])
        
        # Check Stage II
        if self._matches_rule('stage_2', symptoms):
            rule = self.rules['stage_2']
            return ('Stage_II', rule['diagnosis'], rule['prescription'])
        
        # Check Stage I
        if self._matches_rule('stage_1', symptoms):
            rule = self.rules['stage_1']
            return ('Stage_I', rule['diagnosis'], rule['prescription'])
        
        # No malaria detected
        return ('No_Malaria', 
                'No malaria symptoms detected',
                'Continue monitoring symptoms. Consult a doctor if symptoms persist.')
    
    def _matches_rule(self, rule_name: str, symptoms: Dict[str, str]) -> bool:
        """Check if symptoms match a specific rule"""
        rule = self.rules[rule_name]
        
        # Count how many required symptoms are present
        required_count = sum(
            1 for req in rule['required'] 
            if symptoms.get(req) == 'yes'
        )
        
        return required_count >= rule['min_required']

def evaluate_baseline(dataset_path: str):
    """Evaluate baseline system on test dataset"""
    print("🔬 Evaluating Baseline Rule-Based System\n")
    print("="*60)
    
    # Load dataset
    df = pd.read_csv(dataset_path)
    print(f"📊 Dataset: {len(df)} cases loaded\n")
    
    # Initialize system
    system = MalariaExpertSystem()
    
    # Evaluate each case
    results = []
    correct = 0
    
    symptom_cols = [
        'headache', 'cough', 'body_pain', 'poor_appetite', 'rash',
        'sweating', 'chest_pain', 'symptoms_less_1week', 'abdominal_pain',
        'constipation', 'restlessness', 'diarrhea', 'dizziness',
        'semi_closed_eyes', 'exhaustion', 'symptoms_over_1week',
        'hallucination', 'back_pain', 'blurry_vision'
    ]
    
    for idx, row in df.iterrows():
        # Extract symptoms
        symptoms = {col: row[col] for col in symptom_cols if col in row}
        
        # Diagnose
        predicted_stage, diagnosis, prescription = system.diagnose(symptoms)
        
        # Compare with expected
        expected_stage = row.get('severity_stage', 'Unknown')
        is_correct = (predicted_stage == expected_stage)
        
        if is_correct:
            correct += 1
        
        results.append({
            'case_id': row['case_id'],
            'expected': expected_stage,
            'predicted': predicted_stage,
            'correct': is_correct,
            'diagnosis': diagnosis
        })
    
    # Calculate metrics
    accuracy = (correct / len(df)) * 100
    
    # Print results
    results_df = pd.DataFrame(results)
    
    print(f"✅ Evaluation Complete")
    print(f"\n📈 Overall Accuracy: {accuracy:.2f}%")
    print(f"   Correct: {correct}/{len(df)}")
    
    print(f"\n📊 Confusion Matrix:")
    confusion = pd.crosstab(
        results_df['expected'], 
        results_df['predicted'],
        margins=True
    )
    print(confusion)
    
    # Save results
    output_path = 'data/processed/baseline_results.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\n💾 Results saved to: {output_path}")
    
    return results_df, accuracy

if __name__ == "__main__":
    import sys
    
    # Check if combined dataset exists
    combined_path = 'data/processed/malaria_combined_dataset.csv'
    synthetic_path = 'data/processed/malaria_evaluation_dataset.csv'
    
if os.path.exists(combined_path):
        print("🎯 Using COMBINED dataset (Kaggle + Synthetic)\n")
        dataset_path = combined_path
else:
        print("⚠️  Using SYNTHETIC dataset only")
        print("   Download Kaggle data for better evaluation!\n")
        dataset_path = synthetic_path
    
    # Evaluate
results, accuracy = evaluate_baseline(dataset_path)
