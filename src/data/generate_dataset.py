"""
Download and merge Kaggle malaria symptom dataset with our generated data
"""
import pandas as pd
import os

def download_instructions():
    """Print instructions for manual download"""
    print("📥 DATASET DOWNLOAD INSTRUCTIONS")
    print("="*60)
    print("\n1. Go to: https://www.kaggle.com/datasets/programmer3/malaria-diagnosis-dataset")
    print("\n2. Click 'Download' button (requires Kaggle account)")
    print("\n3. Extract the ZIP file")
    print("\n4. Move 'Malaria_Dataset.csv' to: data/raw/")
    print("\n5. Run this script again")
    print("\n" + "="*60)

def load_and_preprocess_kaggle_data():
    """Load Kaggle malaria dataset and standardize to our format"""
    kaggle_file = 'data/raw/Malaria_Dataset.csv'
    
    if not os.path.exists(kaggle_file):
        download_instructions()
        return None
    
    print("✅ Loading Kaggle dataset...")
    df = pd.read_csv(kaggle_file)
    
    print(f"  → Original shape: {df.shape}")
    print(f"  → Columns: {list(df.columns)}")
    
    # Map Kaggle columns to our symptom format
    symptom_mapping = {
        'Fever': 'fever',
        'Headache': 'headache',
        'Coughing': 'cough',
        'Abdominal_Pain': 'abdominal_pain',
        'General_Body_Malaise': 'body_pain',
        'Dizziness': 'dizziness',
        'Vomiting': 'vomiting',
        'Confusion': 'confusion',
        'Backache': 'back_pain',
        'Chest_Pain': 'chest_pain',
        'Joint_Pain': 'joint_pain',
    }
    
    # Create standardized dataset
    standardized = pd.DataFrame()
    
    # Add case ID
    standardized['case_id'] = ['KAG_' + str(i).zfill(4) for i in range(len(df))]
    
    # Add demographics
    if 'Age' in df.columns:
        standardized['age'] = df['Age']
    if 'Sex' in df.columns:
        standardized['sex'] = df['Sex'].map({1: 'Male', 0: 'Female'})
    
    # Add symptoms (convert to yes/no format)
    for kaggle_col, our_col in symptom_mapping.items():
        if kaggle_col in df.columns:
            standardized[our_col] = df[kaggle_col].map({1: 'yes', 0: 'no'})
    
    # Add missing symptoms from our schema (mark as 'unknown')
    our_symptoms = [
        'headache', 'cough', 'body_pain', 'poor_appetite', 'rash',
        'sweating', 'chest_pain', 'symptoms_less_1week', 'abdominal_pain',
        'constipation', 'restlessness', 'diarrhea', 'dizziness',
        'semi_closed_eyes', 'exhaustion', 'symptoms_over_1week',
        'hallucination', 'back_pain', 'blurry_vision'
    ]
    
    for symptom in our_symptoms:
        if symptom not in standardized.columns:
            standardized[symptom] = 'no'  # Conservative default
    
    # Add target (malaria diagnosis)
    if 'Target' in df.columns:
        standardized['malaria_positive'] = df['Target'].map({1: 'yes', 0: 'no'})
    
    # Infer severity stage based on symptom count
    symptom_cols = [col for col in standardized.columns if col in our_symptoms]
    standardized['symptom_count'] = standardized[symptom_cols].apply(
        lambda row: (row == 'yes').sum(), axis=1
    )
    
    def classify_severity(row):
        if row['malaria_positive'] == 'no':
            return 'No_Malaria'
        elif row['symptom_count'] <= 5:
            return 'Stage_I'
        elif row['symptom_count'] <= 10:
            return 'Stage_II'
        else:
            return 'Critical'
    
    standardized['severity_stage'] = standardized.apply(classify_severity, axis=1)
    
    # Add default values for missing fields
    standardized['genotype'] = 'AA'  # Default
    standardized['blood_type'] = 'O+'  # Default
    standardized['pregnant'] = 'no'  # Default
    standardized['rdt_result'] = standardized['malaria_positive'].map({
        'yes': 'positive', 'no': 'negative'
    })
    standardized['microscopy_result'] = 'not_done'
    
    print(f"\n✅ Standardized dataset shape: {standardized.shape}")
    print(f"  → Severity distribution:")
    print(standardized['severity_stage'].value_counts())
    
    return standardized

def merge_datasets():
    """Merge our synthetic data with Kaggle data"""
    # Load our generated data
    our_data_path = 'data/processed/malaria_evaluation_dataset.csv'
    
    if not os.path.exists(our_data_path):
        print("❌ Please run generate_dataset.py first!")
        return
    
    our_data = pd.read_csv(our_data_path)
    print(f"✅ Our synthetic dataset: {len(our_data)} cases")
    
    # Load Kaggle data
    kaggle_data = load_and_preprocess_kaggle_data()
    
    if kaggle_data is None:
        print("\n⚠️  Using only synthetic dataset for now")
        return
    
    # Combine datasets
    print("\n🔄 Merging datasets...")
    
    # Ensure both have same columns
    all_columns = set(our_data.columns) | set(kaggle_data.columns)
    
    for col in all_columns:
        if col not in our_data.columns:
            our_data[col] = 'no' if col in ['headache', 'cough'] else ''
        if col not in kaggle_data.columns:
            kaggle_data[col] = 'no' if col in ['headache', 'cough'] else ''
    
    # Align column order
    common_cols = sorted(all_columns)
    our_data = our_data[common_cols]
    kaggle_data = kaggle_data[common_cols]
    
    # Merge
    combined = pd.concat([our_data, kaggle_data], ignore_index=True)
    
    # Save merged dataset
    output_path = 'data/processed/malaria_combined_dataset.csv'
    combined.to_csv(output_path, index=False)
    
    print(f"\n✅ Combined dataset saved: {output_path}")
    print(f"  → Total cases: {len(combined)}")
    print(f"  → Synthetic: {len(our_data)}")
    print(f"  → Kaggle: {len(kaggle_data)}")
    print(f"\n📊 Final severity distribution:")
    print(combined['severity_stage'].value_counts())

if __name__ == "__main__":
    print("🚀 Malaria Dataset Integration Tool\n")
    merge_datasets()
