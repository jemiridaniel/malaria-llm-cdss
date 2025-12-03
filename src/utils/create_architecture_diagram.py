"""
Create professional architecture diagram for the malaria diagnosis system
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_architecture_diagram():
    """Generate system architecture diagram"""
    
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Color scheme
    colors = {
        'input': '#3498db',      # Blue
        'process': '#2ecc71',    # Green
        'ai': '#e74c3c',         # Red
        'output': '#f39c12',     # Orange
        'data': '#9b59b6'        # Purple
    }
    
    # Title
    ax.text(8, 9.5, 'Malaria LLM-Enhanced Clinical Decision Support System', 
            ha='center', fontsize=18, fontweight='bold')
    ax.text(8, 9.0, 'Hybrid Architecture: Rule-Based Classification + LLM Reasoning',
            ha='center', fontsize=12, style='italic', color='gray')
    
    # Layer 1: Input Layer (Bottom)
    y_input = 0.5
    
    # Patient Input
    box1 = FancyBboxPatch((0.5, y_input), 2.5, 1.2, 
                          boxstyle="round,pad=0.1", 
                          facecolor=colors['input'], 
                          edgecolor='black', linewidth=2)
    ax.add_patch(box1)
    ax.text(1.75, y_input + 0.85, 'Patient Input', ha='center', fontsize=11, fontweight='bold', color='white')
    ax.text(1.75, y_input + 0.5, '• 19 Symptoms', ha='center', fontsize=8, color='white')
    ax.text(1.75, y_input + 0.3, '• Demographics', ha='center', fontsize=8, color='white')
    ax.text(1.75, y_input + 0.1, '• Test Results', ha='center', fontsize=8, color='white')
    
    # Layer 2: Data Processing
    y_process = 2.5
    
    # Symptom Collection
    box2 = FancyBboxPatch((0.5, y_process), 2.5, 1.2,
                          boxstyle="round,pad=0.1",
                          facecolor=colors['process'],
                          edgecolor='black', linewidth=2)
    ax.add_patch(box2)
    ax.text(1.75, y_process + 0.85, 'Symptom Collection', ha='center', fontsize=11, fontweight='bold', color='white')
    ax.text(1.75, y_process + 0.5, 'Structured Input', ha='center', fontsize=8, color='white')
    ax.text(1.75, y_process + 0.2, 'Data Validation', ha='center', fontsize=8, color='white')
    
    # Arrow: Input → Collection
    arrow1 = FancyArrowPatch((1.75, y_input + 1.2), (1.75, y_process),
                            arrowstyle='->', mutation_scale=30, 
                            linewidth=2.5, color='black')
    ax.add_patch(arrow1)
    
    # Layer 3: Classification Layer
    y_classify = 4.5
    
    # Rule-Based Classifier
    box3 = FancyBboxPatch((0.2, y_classify), 3.2, 1.5,
                          boxstyle="round,pad=0.1",
                          facecolor=colors['ai'],
                          edgecolor='black', linewidth=2)
    ax.add_patch(box3)
    ax.text(1.8, y_classify + 1.15, 'Rule-Based Classifier', ha='center', fontsize=11, fontweight='bold', color='white')
    ax.text(1.8, y_classify + 0.8, 'WHO Guidelines', ha='center', fontsize=8, color='white')
    ax.text(1.8, y_classify + 0.5, '• Critical: Hallucination/Semi-conscious', ha='center', fontsize=7, color='white')
    ax.text(1.8, y_classify + 0.3, '• Stage II: 7-10 symptoms + duration', ha='center', fontsize=7, color='white')
    ax.text(1.8, y_classify + 0.1, '• Stage I: 3-6 symptoms', ha='center', fontsize=7, color='white')
    
    # Arrow: Collection → Classifier
    arrow2 = FancyArrowPatch((1.75, y_process + 1.2), (1.8, y_classify),
                            arrowstyle='->', mutation_scale=30,
                            linewidth=2.5, color='black')
    ax.add_patch(arrow2)
    
    # Layer 4: LLM Layer
    y_llm = 4.5
    
    # LLM Reasoning
    box4 = FancyBboxPatch((4.0, y_llm), 3.5, 1.5,
                          boxstyle="round,pad=0.1",
                          facecolor=colors['ai'],
                          edgecolor='black', linewidth=2)
    ax.add_patch(box4)
    ax.text(5.75, y_llm + 1.15, 'LLM Reasoning Layer', ha='center', fontsize=11, fontweight='bold', color='white')
    ax.text(5.75, y_llm + 0.85, 'Llama 3.1 8B (Ollama)', ha='center', fontsize=9, style='italic', color='white')
    ax.text(5.75, y_llm + 0.5, '• Clinical Explanation', ha='center', fontsize=8, color='white')
    ax.text(5.75, y_llm + 0.3, '• Treatment Rationale', ha='center', fontsize=8, color='white')
    ax.text(5.75, y_llm + 0.1, '• Confidence Scoring', ha='center', fontsize=8, color='white')
    
    # Arrow: Classifier → LLM (bidirectional)
    arrow3 = FancyArrowPatch((3.4, y_classify + 0.75), (4.0, y_llm + 0.75),
                            arrowstyle='<->', mutation_scale=30,
                            linewidth=2.5, color='purple')
    ax.add_patch(arrow3)
    ax.text(3.7, y_classify + 1.0, 'Classification\n+ Context', ha='center', fontsize=7, color='purple', fontweight='bold')
    
    # Knowledge Base
    y_kb = 6.5
    box5 = FancyBboxPatch((8.5, y_kb), 3.0, 1.0,
                          boxstyle="round,pad=0.1",
                          facecolor=colors['data'],
                          edgecolor='black', linewidth=2)
    ax.add_patch(box5)
    ax.text(10, y_kb + 0.7, 'Knowledge Base', ha='center', fontsize=11, fontweight='bold', color='white')
    ax.text(10, y_kb + 0.4, 'WHO Malaria Guidelines', ha='center', fontsize=8, color='white')
    ax.text(10, y_kb + 0.15, 'Treatment Protocols', ha='center', fontsize=8, color='white')
    
    # Arrow: KB → LLM
    arrow4 = FancyArrowPatch((8.5, y_kb + 0.5), (7.5, y_llm + 1.0),
                            arrowstyle='->', mutation_scale=25,
                            linewidth=2, color='purple', linestyle='dashed')
    ax.add_patch(arrow4)
    ax.text(8.2, y_kb - 0.3, 'Guidelines\nRetrieval', ha='center', fontsize=7, color='purple')
    
    # Layer 5: Output Layer
    y_output = 7.5
    
    # Diagnosis Output
    box6 = FancyBboxPatch((1.5, y_output), 2.8, 1.3,
                          boxstyle="round,pad=0.1",
                          facecolor=colors['output'],
                          edgecolor='black', linewidth=2)
    ax.add_patch(box6)
    ax.text(2.9, y_output + 0.95, 'Diagnosis Output', ha='center', fontsize=11, fontweight='bold', color='white')
    ax.text(2.9, y_output + 0.6, '• Severity Classification', ha='center', fontsize=8, color='white')
    ax.text(2.9, y_output + 0.4, '• Confidence Score', ha='center', fontsize=8, color='white')
    ax.text(2.9, y_output + 0.2, '• Clinical Reasoning', ha='center', fontsize=8, color='white')
    
    # Prescription Output
    box7 = FancyBboxPatch((5.0, y_output), 2.8, 1.3,
                          boxstyle="round,pad=0.1",
                          facecolor=colors['output'],
                          edgecolor='black', linewidth=2)
    ax.add_patch(box7)
    ax.text(6.4, y_output + 0.95, 'Treatment Plan', ha='center', fontsize=11, fontweight='bold', color='white')
    ax.text(6.4, y_output + 0.6, '• Medication', ha='center', fontsize=8, color='white')
    ax.text(6.4, y_output + 0.4, '• Care Instructions', ha='center', fontsize=8, color='white')
    ax.text(6.4, y_output + 0.2, '• Escalation Flags', ha='center', fontsize=8, color='white')
    
    # Arrows: LLM → Outputs
    arrow5 = FancyArrowPatch((3.4, y_classify + 1.5), (2.9, y_output),
                            arrowstyle='->', mutation_scale=30,
                            linewidth=2.5, color='black')
    ax.add_patch(arrow5)
    
    arrow6 = FancyArrowPatch((5.75, y_llm + 1.5), (6.4, y_output),
                            arrowstyle='->', mutation_scale=30,
                            linewidth=2.5, color='black')
    ax.add_patch(arrow6)
    
    # Performance Metrics Box
    box8 = FancyBboxPatch((12.0, 0.5), 3.5, 3.0,
                          boxstyle="round,pad=0.15",
                          facecolor='#ecf0f1',
                          edgecolor='black', linewidth=2)
    ax.add_patch(box8)
    ax.text(13.75, 3.2, 'System Performance', ha='center', fontsize=11, fontweight='bold')
    ax.text(13.75, 2.8, '━━━━━━━━━━━━━━━━', ha='center', fontsize=10, color='gray')
    ax.text(13.75, 2.45, '✓ Overall Accuracy: 87.22%', ha='center', fontsize=9)
    ax.text(13.75, 2.15, '✓ Critical Detection: 100%', ha='center', fontsize=9)
    ax.text(13.75, 1.85, '✓ Stage I Detection: 98.9%', ha='center', fontsize=9)
    ax.text(13.75, 1.55, '✓ Processing Time: 6.3s', ha='center', fontsize=9)
    ax.text(13.75, 1.25, '✓ Improvement: +56.6%', ha='center', fontsize=9, color='green', fontweight='bold')
    ax.text(13.75, 0.85, '📊 Dataset: 1,682 cases', ha='center', fontsize=8, style='italic')
    
    # Tech Stack Box
    box9 = FancyBboxPatch((12.0, 4.0), 3.5, 2.5,
                          boxstyle="round,pad=0.15",
                          facecolor='#ecf0f1',
                          edgecolor='black', linewidth=2)
    ax.add_patch(box9)
    ax.text(13.75, 6.2, 'Technology Stack', ha='center', fontsize=11, fontweight='bold')
    ax.text(13.75, 5.8, '━━━━━━━━━━━━━━━━', ha='center', fontsize=10, color='gray')
    ax.text(13.75, 5.5, '🐍 Python 3.11', ha='center', fontsize=9)
    ax.text(13.75, 5.2, '🤖 Llama 3.1 8B (Ollama)', ha='center', fontsize=9)
    ax.text(13.75, 4.9, '📊 Pandas, NumPy, Scikit-learn', ha='center', fontsize=9)
    ax.text(13.75, 4.6, '🌐 Flask API', ha='center', fontsize=9)
    ax.text(13.75, 4.3, '💾 PostgreSQL + pgvector', ha='center', fontsize=9)
    
    # Legend
    legend_y = 8.5
    ax.text(12.5, legend_y + 0.3, 'Legend:', fontsize=10, fontweight='bold')
    
    # Legend items
    legend_items = [
        (colors['input'], 'Input Layer'),
        (colors['process'], 'Processing'),
        (colors['ai'], 'AI/ML Components'),
        (colors['output'], 'Output Layer'),
        (colors['data'], 'Knowledge Base')
    ]
    
    for i, (color, label) in enumerate(legend_items):
        y_pos = legend_y - (i * 0.25)
        rect = mpatches.Rectangle((12.5, y_pos - 0.1), 0.3, 0.15, 
                                  facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        ax.text(12.9, y_pos, label, fontsize=8, va='center')
    
    # Footer
    ax.text(8, 0.2, '© 2025 Daniel Jemiri | GitHub: @jemiridaniel/malaria-llm-cdss',
            ha='center', fontsize=8, style='italic', color='gray')
    
    plt.tight_layout()
    plt.savefig('results/figures/system_architecture.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✅ Architecture diagram saved: results/figures/system_architecture.png")
    plt.close()

if __name__ == "__main__":
    create_architecture_diagram()
    print("🎨 Architecture diagram generation complete!")