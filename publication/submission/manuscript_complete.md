# An LLM-Enhanced Clinical Decision Support System for Malaria Diagnosis in Resource-Limited Settings: A Hybrid Approach

**Running Title:** LLM-Enhanced Malaria Diagnosis System

**Author:**  
Daniel Jemiri, MS  
The University of Texas at Dallas, Richardson, TX, United States  
ORCID: https://orcid.org/0009-0005-9338-7682

**Corresponding Author:**  
Daniel Jemiri  
Email: Daniel.Jemiri@UTDallas.edu

---

## ABSTRACT

**Background:** Malaria remains a significant global health challenge, causing approximately 600,000 deaths annually, predominantly in sub-Saharan Africa. Traditional rule-based expert systems for malaria diagnosis have limitations in handling symptom variability and providing clinical reasoning to support healthcare worker decisions.

**Objective:** To develop and evaluate an LLM-enhanced clinical decision support system for malaria diagnosis that combines deterministic classification rules with large language model reasoning capabilities to improve diagnostic accuracy and explainability in resource-limited settings.

**Methods:** We developed a hybrid system integrating a rule-based classifier with Llama 3.1 8B for natural language explanation generation. The rule-based component implements WHO malaria severity classification guidelines, while the LLM provides clinical reasoning, treatment recommendations, and confidence scoring. The system was evaluated on 1,682 malaria cases (1,622 real-world cases from the Kaggle Malaria Diagnosis Dataset plus 60 synthetic cases) spanning four severity levels: No Malaria, Stage I (uncomplicated), Stage II (moderate), and Critical. Performance was compared against the original rule-based baseline using accuracy, precision, recall, and F1-score metrics.

**Results:** The LLM-enhanced hybrid system achieved 87.22% diagnostic accuracy (1,467/1,682 correct classifications) compared to 30.62% for the baseline rule-based system, representing a 56.60 percentage point improvement (relative improvement of 185%). The system demonstrated 100% accuracy (20/20) in detecting critical malaria cases, 98.9% accuracy (1,077/1,089) for Stage I cases, and 77.4% accuracy (352/455) for No Malaria cases. Average processing time was 6.3 seconds per diagnosis with an average LLM confidence score of 85%. The hybrid approach maintained the baseline's perfect critical case detection while dramatically improving Stage I detection from 1.8% to 98.9%.

**Conclusions:** Integration of large language models with rule-based medical decision support systems significantly improves diagnostic accuracy while adding natural language explainability. This hybrid approach demonstrates promise for deployment in resource-limited healthcare settings where specialist physicians are scarce. The system's offline capability, zero API costs, and privacy-preserving architecture address key barriers to AI adoption in low- and middle-income countries. Future work should include prospective clinical validation and expansion to differential diagnosis of other febrile illnesses.

**Keywords:** malaria diagnosis; clinical decision support systems; large language models; artificial intelligence; resource-limited settings; explainable AI; hybrid systems; global health

---

## INTRODUCTION

[Your introduction from results/paper_draft.md]

---

## METHODS

[Your methodology section]

---

## RESULTS

[Your results section with tables and figures]

---

## DISCUSSION

[Your discussion section]

---

## CONCLUSION

[Your conclusion]

---

## Data Availability Statement

The dataset used in this study is publicly available:
- Synthetic dataset: Available at https://github.com/jemiridaniel/malaria-llm-cdss/data
- Real-world data: Kaggle Malaria Diagnosis Dataset (https://kaggle.com/datasets/programmer3/malaria-diagnosis-dataset)
- Source code: https://github.com/jemiridaniel/malaria-llm-cdss
- Evaluation results: Included in supplementary materials

All code, data processing scripts, and evaluation notebooks are openly available under the MIT License to facilitate reproducibility and future research.

---

## Ethics Statement

This computational study utilized publicly available datasets and synthetic data. No human subjects were directly involved in data collection. The study does not require Institutional Review Board (IRB) approval as it constitutes a retrospective analysis of de-identified datasets. 

The system is designed and presented as clinical decision support, not autonomous diagnosis. Implementation includes appropriate safety warnings and human-in-the-loop requirements for clinical deployment. Future prospective clinical validation studies will require appropriate ethical approvals.

---

## Acknowledgments

We acknowledge the open-source community for Ollama and Meta AI for Llama 3.1. We thank the contributors to the Kaggle Malaria Diagnosis Dataset for making their data publicly available. The original undergraduate system was developed at Federal University of Technology Akure (FUTA), Nigeria, in 2020.

---

## Conflicts of Interest

None declared. This research was conducted independently without external funding or commercial interests.

---

## REFERENCES

[30 references from references.bib - to be formatted in AMA style]

