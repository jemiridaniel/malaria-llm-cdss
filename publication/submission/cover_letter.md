cat > publication/submission/cover_letter.md << 'EOF'
**Cover Letter**

Date: December 3, 2025

**To:** JMIR Medical Informatics Editorial Board

**Re:** Manuscript Submission - Original Research

**Title:** An LLM-Enhanced Clinical Decision Support System for Malaria Diagnosis in Resource-Limited Settings: A Hybrid Approach

Dear Editors,

I am pleased to submit our manuscript for consideration for publication in JMIR Medical Informatics. This work presents a novel hybrid approach combining rule-based classification with large language models (LLMs) for malaria diagnosis, achieving 87.22% diagnostic accuracy—a 56.60 percentage point improvement over traditional rule-based systems.

## **Significance and Innovation**

**Clinical Impact:** Malaria causes approximately 600,000 deaths annually, predominantly in sub-Saharan Africa where access to specialist physicians is limited. Our system addresses this critical gap by providing accurate, explainable diagnostic support suitable for deployment in rural health posts and resource-limited settings.

**Technical Innovation:** While prior LLM healthcare research has focused on image analysis or general medical question-answering, our work demonstrates:
- First implementation of a hybrid rule-based + LLM system for malaria diagnosis
- Novel approach using free, offline LLM deployment (Ollama/Llama 3.1)
- Privacy-preserving architecture with no external API dependencies
- 100% detection rate for critical life-threatening cases

**Validation:** Evaluated on 1,682 real-world and synthetic cases across four severity stages, demonstrating robust performance suitable for clinical deployment.

## **Relevance to JMIR Medical Informatics**

This manuscript aligns with your journal's focus on:
1. **AI in Healthcare:** Novel application of LLMs in clinical decision support
2. **Digital Health in LMICs:** Designed specifically for resource-limited settings
3. **Explainable AI:** Natural language reasoning enhances clinician trust
4. **Implementation Science:** Practical, deployable system with clear clinical utility

## **Key Findings**

- **87.22% overall diagnostic accuracy** (vs 30.62% baseline)
- **100% critical case detection** (perfect safety record)
- **98.9% Stage I malaria detection** (enabling early intervention)
- **6.3 seconds processing time** (suitable for clinical workflow)
- **Zero API costs** (sustainable for low-resource deployment)

## **Ethical Considerations**

This research:
- Uses publicly available datasets (Kaggle + synthetic data)
- Does not involve human subjects directly (computational study)
- Presents system as decision support, not autonomous diagnosis
- Includes comprehensive discussion of limitations and safety measures

## **Data and Code Availability**

In accordance with JMIR policies:
- Full source code: https://github.com/jemiridaniel/malaria-llm-cdss
- Dataset: Kaggle Malaria Diagnosis Dataset (publicly available)
- Synthetic data: Included in repository
- All evaluation results: Available in supplementary materials

## **Author Contributions**

Daniel Jemiri (sole author):
- Conceptualization, system design, implementation
- Data collection and analysis
- Manuscript writing and revision
- Original undergraduate system (2020) and current enhancement (2025)

## **Conflicts of Interest**

No conflicts of interest to declare. This work was conducted independently without industry funding or commercial interests.

## **Prior Presentations**

This work has not been previously published or submitted elsewhere. It represents a substantial evolution of an undergraduate project completed at Federal University of Technology Akure, Nigeria (2020), with entirely new methodology, evaluation, and analysis.

## **Manuscript Statistics**

- Word count: ~4,500 words (excluding references)
- Tables: 2
- Figures: 2
- Supplementary materials: 3 appendices

## **Requested Reviewers**

We suggest reviewers with expertise in:
1. LLMs in healthcare
2. Clinical decision support systems
3. Tropical medicine/malaria diagnosis
4. Digital health in low-resource settings

(Specific names available upon request)

## **Why JMIR Medical Informatics?**

Your journal's open-access model ensures our findings reach healthcare workers in malaria-endemic regions who would benefit most from this technology. The rapid review process (8-12 weeks) aligns with our goal of timely dissemination to inform clinical practice.

We believe this manuscript will be of significant interest to your readership, particularly those working in AI-enhanced healthcare, global health informatics, and clinical decision support systems.

Thank you for considering our manuscript. We look forward to your feedback and are prepared to address reviewer comments promptly.

Sincerely,

**Daniel Jemiri**  
PhD Student, Artificial Intelligence & Information Science  
The University of Texas at Dallas  
Dallas, Texas, United States  
Email: [Your Email](Daniel.Jemiri@UTDallas.edu)  
ORCID: [[Your ORCID](https://orcid.org/0009-0005-9338-7682)]  
GitHub: @jemiridaniel

**Corresponding Author:** Same as above

