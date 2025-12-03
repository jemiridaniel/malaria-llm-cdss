
"""
Comparative Analysis: Baseline vs LLM-Enhanced Systems
Generates tables and statistics for journal paper
"""
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import json

def load_results():
    """Load all evaluation results"""
    results = {}
    
    # Baseline (rule-based)
    baseline_path = 'data/processed/baseline_results.csv'
    if pd.io.common.file_exists(baseline_path):
        results['baseline'] = pd.read_csv(baseline_path)
        print("✅ Loaded baseline results")
    else:
        print("⚠️  Baseline results not found")
    
    # LLM v1 (pure Ollama)
    llm_v1_path = 'data/processed/llm_ollama_results.csv'
    if pd.io.common.file_exists(llm_v1_path):
        results['llm_v1'] = pd.read_csv(llm_v1_path)
        print("✅ Loaded LLM v1 results")
    
    # LLM v2 (optimized prompts)
    llm_v2_path = 'data/processed/llm_ollama_v2_results.csv'
    if pd.io.common.file_exists(llm_v2_path):
        results['llm_v2'] = pd.read_csv(llm_v2_path)
        print("✅ Loaded LLM v2 results")
    
    # Hybrid (rules + LLM)
    hybrid_path = 'data/processed/hybrid_results.csv'
    if pd.io.common.file_exists(hybrid_path):
        results['hybrid'] = pd.read_csv(hybrid_path)
        print("✅ Loaded hybrid results")
    else:
        print("⚠️  Hybrid results not found - run full evaluation first")
    
    return results

def calculate_metrics(df):
    """Calculate comprehensive metrics"""
    y_true = df['expected']
    y_pred = df['predicted']
    
    # Overall accuracy
    accuracy = (df['correct'].sum() / len(df)) * 100
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, 
        labels=['No_Malaria', 'Stage_I', 'Stage_II', 'Critical'],
        zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(
        y_true, y_pred,
        labels=['No_Malaria', 'Stage_I', 'Stage_II', 'Critical']
    )
    
    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision.mean() * 100,
        'recall_macro': recall.mean() * 100,
        'f1_macro': f1.mean() * 100,
        'per_class': {
            'No_Malaria': {'precision': precision[0]*100, 'recall': recall[0]*100, 'f1': f1[0]*100, 'support': support[0]},
            'Stage_I': {'precision': precision[1]*100, 'recall': recall[1]*100, 'f1': f1[1]*100, 'support': support[1]},
            'Stage_II': {'precision': precision[2]*100, 'recall': recall[2]*100, 'f1': f1[2]*100, 'support': support[2]},
            'Critical': {'precision': precision[3]*100, 'recall': recall[3]*100, 'f1': f1[3]*100, 'support': support[3]}
        },
        'confusion_matrix': cm.tolist()
    }
    
    return metrics

def generate_comparison_table(results):
    """Generate comparison table for paper"""
    print("\n" + "="*80)
    print("📊 SYSTEM PERFORMANCE COMPARISON")
    print("="*80 + "\n")
    
    comparison = []
    
    for system_name, df in results.items():
        if df is None or len(df) == 0:
            continue
        
        metrics = calculate_metrics(df)
        
        comparison.append({
            'System': system_name.replace('_', ' ').title(),
            'Accuracy (%)': f"{metrics['accuracy']:.2f}",
            'Precision (%)': f"{metrics['precision_macro']:.2f}",
            'Recall (%)': f"{metrics['recall_macro']:.2f}",
            'F1-Score (%)': f"{metrics['f1_macro']:.2f}",
            'Cases': len(df)
        })
    
    comp_df = pd.DataFrame(comparison)
    print(comp_df.to_string(index=False))
    
    # Save for LaTeX
    latex_table = comp_df.to_latex(index=False, float_format="%.2f")
    with open('docs/comparison_table.tex', 'w') as f:
        f.write(latex_table)
    print("\n💾 LaTeX table saved to: docs/comparison_table.tex")
    
    return comp_df

def generate_per_stage_analysis(results):
    """Per-stage performance breakdown"""
    print("\n" + "="*80)
    print("📉 PER-STAGE PERFORMANCE ANALYSIS")
    print("="*80 + "\n")
    
    stages = ['No_Malaria', 'Stage_I', 'Stage_II', 'Critical']
    
    for stage in stages:
        print(f"\n{stage}:")
        print("-" * 60)
        
        for system_name, df in results.items():
            if df is None or len(df) == 0:
                continue
            
            stage_df = df[df['expected'] == stage]
            if len(stage_df) == 0:
                continue
            
            correct = stage_df['correct'].sum()
            total = len(stage_df)
            accuracy = (correct / total) * 100
            
            print(f"  {system_name:15s}: {accuracy:5.1f}% ({correct}/{total})")

def generate_improvement_summary(results):
    """Calculate improvement over baseline"""
    print("\n" + "="*80)
    print("📈 IMPROVEMENT OVER BASELINE")
    print("="*80 + "\n")
    
    if 'baseline' not in results or results['baseline'] is None:
        print("⚠️  Baseline results not available")
        return
    
    baseline_acc = (results['baseline']['correct'].sum() / len(results['baseline'])) * 100
    
    print(f"Baseline Accuracy: {baseline_acc:.2f}%\n")
    
    for system_name, df in results.items():
        if system_name == 'baseline' or df is None or len(df) == 0:
            continue
        
        system_acc = (df['correct'].sum() / len(df)) * 100
        improvement = system_acc - baseline_acc
        relative_improvement = (improvement / baseline_acc) * 100
        
        print(f"{system_name.replace('_', ' ').title():20s}: "
              f"{system_acc:5.2f}% (Δ {improvement:+5.2f}%, "
              f"{relative_improvement:+5.1f}% relative)")

def generate_confusion_matrices(results):
    """Generate confusion matrices for all systems"""
    print("\n" + "="*80)
    print("🔢 CONFUSION MATRICES")
    print("="*80 + "\n")
    
    for system_name, df in results.items():
        if df is None or len(df) == 0:
            continue
        
        print(f"\n{system_name.upper()}:")
        print("-" * 60)
        
        cm = pd.crosstab(
            df['expected'], 
            df['predicted'],
            rownames=['Actual'],
            colnames=['Predicted'],
            margins=True
        )
        print(cm)

def generate_statistical_summary(results):
    """Generate detailed statistical summary for methods section"""
    print("\n" + "="*80)
    print("📊 STATISTICAL SUMMARY FOR PAPER")
    print("="*80 + "\n")
    
    summary = {}
    
    for system_name, df in results.items():
        if df is None or len(df) == 0:
            continue
        
        metrics = calculate_metrics(df)
        
        summary[system_name] = {
            'total_cases': len(df),
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision_macro'],
            'recall': metrics['recall_macro'],
            'f1_score': metrics['f1_macro'],
            'per_stage': metrics['per_class']
        }
        
        # Add processing time if available
        if 'processing_time' in df.columns:
            summary[system_name]['avg_time_seconds'] = df['processing_time'].mean()
            summary[system_name]['total_time_minutes'] = df['processing_time'].sum() / 60
        
        # Add confidence if available
        if 'confidence' in df.columns:
            summary[system_name]['avg_confidence'] = df['confidence'].mean()
    
    # Save as JSON
    with open('docs/statistical_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("Saved to: docs/statistical_summary.json")
    print("\nSummary:")
    print(json.dumps(summary, indent=2))

def main():
    """Run complete analysis"""
    print("\n🔬 COMPARATIVE ANALYSIS TOOL")
    print("="*80 + "\n")
    
    # Create docs folder
    import os
    os.makedirs('docs', exist_ok=True)
    
    # Load all results
    results = load_results()
    
    if len(results) == 0:
        print("\n❌ No results files found. Run evaluations first.")
        return
    
    # Generate all analyses
    generate_comparison_table(results)
    generate_per_stage_analysis(results)
    generate_improvement_summary(results)
    generate_confusion_matrices(results)
    generate_statistical_summary(results)
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - docs/comparison_table.tex (for LaTeX paper)")
    print("  - docs/statistical_summary.json (for methods section)")
    print("\nUse these results in your journal paper!")

if __name__ == "__main__":
    main()
