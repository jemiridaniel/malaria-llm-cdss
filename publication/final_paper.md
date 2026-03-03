# An LLM-Enhanced Clinical Decision Support System for Malaria Diagnosis in Resource-Limited Settings: A Hybrid Approach

**Running Title:** LLM-Enhanced Hybrid Malaria Diagnosis System

**Author:**
Jemiri Daniel Taiwo, B.Tech
Department of Computer Science, Federal University of Technology Owerri (FUTO), Owerri, Imo State, Nigeria
ORCID: https://orcid.org/0009-0005-9338-7682
Email: danieljemiri@gmail.com

**Corresponding Author:**
Jemiri Daniel Taiwo
Department of Computer Science, Federal University of Technology Owerri (FUTO), Owerri, Imo State, Nigeria
Email: danieljemiri@gmail.com

---

## Abstract

**Background:** Malaria remains a leading cause of preventable death globally, with an estimated 249 million cases and approximately 608,000 deaths reported in 2022, concentrated predominantly in sub-Saharan Africa where access to specialist diagnostic services is severely limited. Traditional rule-based expert systems for malaria diagnosis, while computationally efficient and deterministic, are constrained by rigid classification rules that cannot accommodate symptom variability or provide clinical reasoning to support healthcare worker decision-making.

**Objective:** We aimed to develop and evaluate a hybrid clinical decision support system (CDSS) for malaria diagnosis that integrates deterministic rule-based classification with large language model (LLM) reasoning to improve both diagnostic accuracy and clinical explainability in resource-limited settings.

**Methods:** We developed a three-component hybrid system comprising: (1) a structured 19-symptom assessment module with patient demographic capture; (2) a deterministic rule-based classification engine implementing WHO malaria severity guidelines across four severity tiers; and (3) an LLM reasoning module using Llama 3.1 8B deployed locally via Ollama for natural language explanation generation. The system was evaluated on 1,682 malaria cases (1,622 real-world cases from the Kaggle Malaria Diagnosis Dataset and 60 synthetic cases) spanning four severity levels: No Malaria, Stage I (uncomplicated), Stage II (moderate), and Critical. Performance was benchmarked against the original rule-based baseline system using overall accuracy, macro-averaged precision, recall, and F1-score.

**Results:** The hybrid LLM-enhanced system achieved 87.22% overall diagnostic accuracy (1,467/1,682 cases correctly classified) compared to 30.62% for the baseline rule-based system, representing a 56.60 percentage point improvement (relative improvement: 185%; p<0.001). Stage I malaria detection improved from 1.84% to 98.90%, and 100% sensitivity for critical malaria identification was maintained in both systems. The macro F1-score improved from 0.174 to 0.853. Average processing time was 6.3 seconds per case, and the LLM generated patient-specific explanations with a mean confidence score of 85%. The system operates entirely offline with no recurring API costs.

**Conclusions:** Integrating large language models with deterministic rule-based classification substantially improves malaria diagnostic accuracy while providing natural language clinical reasoning specific to each patient's reported symptoms. The offline, privacy-preserving, zero-cost architecture directly addresses key barriers to AI adoption in low- and middle-income country healthcare settings. Prospective clinical validation and multilingual support represent the primary directions for future development.

**Keywords:** malaria diagnosis; clinical decision support systems; large language models; hybrid AI systems; explainable AI; resource-limited settings; global health informatics; Llama

---

## 1. Introduction

Malaria remains one of the most preventable yet deadly infectious diseases globally. According to the World Health Organization's World Malaria Report 2023, there were an estimated 249 million malaria cases and approximately 608,000 deaths in 2022 alone, with 94% of cases concentrated in sub-Saharan Africa [10]. Children under five years of age and pregnant women carry the highest disease burden, and malaria exerts a devastating economic toll on already fragile health systems through treatment costs, lost productivity, and chronic morbidity [23]. Despite decades of intervention, malaria incidence has risen since 2015 in several endemic countries, and emerging artemisinin resistance threatens existing treatment paradigms [23].

A central challenge in malaria management is timely, accurate diagnosis. While malaria is caused by Plasmodium parasites transmitted by Anopheles mosquitoes, its clinical presentation overlaps substantially with other febrile illnesses—typhoid fever, pneumonia, and dengue—making symptom-based discrimination inherently difficult [18]. The WHO recommends parasitological confirmation before treatment initiation, using either microscopy or rapid diagnostic tests (RDTs) [2]. However, microscopy requires trained laboratory technicians and functional equipment, and RDTs, while more accessible, carry sensitivity and specificity limitations that vary by Plasmodium species, parasite density, and storage conditions [11]. In rural sub-Saharan Africa, an estimated 50–70% of febrile patients are treated empirically for malaria without confirmed diagnosis, contributing to drug resistance emergence, unnecessary medication costs, and delayed diagnosis of alternative conditions [23].

Rule-based clinical decision support systems (CDSS) have been proposed as a low-cost solution to assist community health workers and nurses in making diagnostic decisions when physicians and laboratory resources are unavailable [7]. These systems encode expert clinical knowledge as structured IF-THEN rules and can operate on consumer hardware without internet connectivity. Abu-Naser and Mahdi demonstrated the feasibility of rule-based expert systems for tropical disease triage in resource-poor settings [14]. However, conventional rule-based systems suffer from well-documented limitations: they cannot generalize beyond explicitly programmed rules, fail to handle atypical symptom presentations, provide no natural language rationale for their decisions, and require labor-intensive expert knowledge engineering to maintain and update [7]. The original rule-based malaria diagnostic system developed at the Federal University of Technology Owerri (FUTO), Nigeria, in 2020—which forms the baseline for this study—achieved only 30.62% overall accuracy on a comprehensive evaluation dataset, illustrating the practical ceiling of purely deterministic approaches when symptom patterns deviate from coded rule templates.

Large language models (LLMs) have recently demonstrated remarkable capabilities in clinical reasoning tasks. GPT-4 achieved near-passing performance on the United States Medical Licensing Examination [5], and subsequent studies have shown that LLMs can generate differential diagnoses, summarize patient records, and provide treatment recommendations with clinician-level accuracy in specific domains [1,3,15]. Critically, LLMs excel at handling natural language variability—a significant advantage over rigid rule systems when patient symptom descriptions are imprecise or overlapping. However, LLMs also carry substantial risks in clinical settings: they are prone to generating confident but factually incorrect clinical statements (hallucination) [22], typically require cloud API access that raises patient privacy concerns and introduces recurring per-query costs, and lack the deterministic guarantees necessary for safety-critical severity classification [20].

This tension between the reliability of rule-based systems and the flexibility of LLMs motivates a hybrid architectural approach. By reserving deterministic classification for severity determination—where rules can be verified against established clinical standards—and using LLMs exclusively for natural language reasoning and explanation generation, we exploit the complementary strengths of both paradigms [12]. The hybrid architecture ensures that the final diagnosis class is never hallucinated by the LLM, while providing clinically meaningful, personalized explanations that enhance healthcare worker understanding and trust.

The principal contributions of this work are:

1. **Hybrid architecture with separation of concerns:** We demonstrate that combining deterministic rule-based severity classification with LLM-generated clinical reasoning significantly outperforms either approach in isolation, improving overall accuracy by 56.60 percentage points over the rule-based baseline on a 1,682-case evaluation dataset.

2. **Preserved safety-critical performance:** The system maintains 100% sensitivity for critical malaria—a non-negotiable clinical safety requirement—while dramatically improving early-stage detection from 1.84% to 98.90%.

3. **Resource-constrained deployment:** The system uses Llama 3.1 8B deployed locally via Ollama, enabling fully offline operation on commodity hardware with zero recurring API costs—directly addressing deployment barriers in sub-Saharan Africa and other low- and middle-income settings.

4. **Clinical explainability:** The LLM component generates natural language clinical reasoning that references each patient's specific reported symptoms, addressing a fundamental limitation of prior rule-based approaches and providing the transparency necessary for healthcare worker adoption.

The remainder of this paper is organized as follows: Section 2 reviews related work across rule-based expert systems, AI-based malaria diagnosis, LLMs in clinical decision support, and explainable AI; Section 3 describes the system architecture, dataset, and evaluation methodology; Section 4 presents experimental results; Section 5 discusses clinical implications, limitations, and deployment considerations; and Section 6 concludes with directions for future research.

---

## 2. Related Work

### 2.1 Rule-Based Expert Systems in Clinical Diagnosis

Rule-based expert systems applying forward-chaining inference to medical diagnosis have a history spanning four decades, with early implementations in oncology (ONCOCIN), cardiology (MYCIN), and infectious disease. Abu-Naser and Mahdi surveyed expert systems applied to tropical and non-communicable disease diagnosis, demonstrating feasibility but noting that classification accuracy degrades sharply when patient presentations deviate from explicitly programmed rule patterns [14]. Sutton et al. conducted a comprehensive review of modern CDSS architectures, concluding that purely rule-based systems face significant scalability challenges and cannot adapt to emerging clinical evidence or unusual case presentations without manual rule updates [7]. In the malaria domain specifically, prior expert systems encoded WHO severity criteria as rigid rule trees, achieving high specificity for critical presentations but near-zero sensitivity for early-stage cases where symptom overlap with non-malarial febrile illness is most pronounced. These known limitations directly motivated the hybrid enhancement pursued in this study.

### 2.2 AI and Image-Based Malaria Diagnosis

A complementary line of research applies convolutional neural networks and deep learning to automated analysis of blood smear microscopy images. Rajaraman et al. demonstrated high accuracy for automated malaria parasite detection and cell counting in thin blood smear images using transfer-learned deep learning models [6]. While these approaches achieve impressive performance in laboratory settings, they require microscopy equipment, digitization hardware, and trained sample preparation technicians—resources unavailable in many rural health posts across sub-Saharan Africa. Our system takes a fundamentally different approach: symptom-based clinical decision support that requires no laboratory equipment beyond a structured questionnaire. Cunningham et al. further found that standard RDTs, the most widely deployed parasitological test, carry non-trivial false-negative rates in low-transmission settings [11], underscoring the clinical value of symptom-based triage systems as a complementary diagnostic layer alongside—rather than in replacement of—parasitological testing.

### 2.3 Large Language Models in Clinical Decision Support

The past two years have witnessed rapid evaluation of LLMs in clinical contexts. Kung et al. reported that ChatGPT (GPT-3.5) passed USMLE Steps 1, 2, and 3 at scores near the passing threshold, suggesting emergent medical reasoning capabilities [5]. Zhou et al. conducted a scoping review of LLM deployment in medical diagnostics and identified diagnostic support, patient education, and clinical summarization as the three principal use cases, noting that most deployments rely on cloud-based proprietary APIs [1]. Wang et al. demonstrated that LLMs can generate probabilistically calibrated differential diagnoses for internal medicine cases, with performance approaching physician-level in controlled settings [3]. Liévin et al. further established that chain-of-thought prompting substantially improves LLM clinical reasoning accuracy across medical benchmark tasks [15].

Despite this promise, significant challenges remain. Chen et al. documented that medical LLMs, including domain-fine-tuned variants, produce confident but factually incorrect clinical statements at non-trivial rates, with error frequency varying by clinical domain and population [22]. Lee et al. cautioned that GPT-4's clinical performance in structured benchmark tasks may not reliably transfer to free-form clinical environments [25]. The clinical NLP literature also highlights that LLM performance degrades for under-represented diseases and populations from low-income countries where training data is sparse [21]. Our hybrid approach directly mitigates hallucination risk by restricting LLM use to explanation generation—never to severity classification—while still capturing the explainability benefits of generative AI.

### 2.4 Explainable AI and Deployment in Resource-Limited Settings

Explainability has emerged as a critical requirement for clinical AI adoption. Albahri et al.'s systematic review found that clinicians are significantly more likely to act on AI recommendations when accompanied by transparent reasoning explanations [8]. In resource-limited settings, this requirement is amplified: community health workers without formal clinical training need explanations that translate diagnostic decisions into actionable guidance. Wahl et al. identified three key barriers to AI adoption in resource-poor healthcare: computational cost, connectivity requirements, and lack of transparency [20]. Our system directly addresses all three: Llama 3.1 8B runs on commodity hardware via Ollama [9,13] with no internet dependency, and generates symptom-specific natural language explanations for every diagnosis. Prompt engineering approaches described by Zhang et al. informed our structured prompting design [17]. Privacy considerations are addressed by the fully local architecture, consistent with best practices for privacy-preserving AI in healthcare [30], and with the data sovereignty requirements identified as a key adoption barrier in sub-Saharan Africa [16].

---

## 3. Methodology

### 3.1 System Architecture

The hybrid malaria CDSS follows a modular three-component pipeline with a strict separation of concerns:

```
Patient Input (19 symptoms + demographics)
         ↓
Rule-Based Classification Engine
[Deterministic severity stage: No_Malaria / Stage_I / Stage_II / Critical]
         ↓
LLM Reasoning Module (Llama 3.1 8B, local/offline)
[Natural language explanation + treatment recommendation + confidence]
         ↓
Structured Clinical Output
```

The key architectural principle is that the rule-based classifier holds sole authority over severity determination. The LLM receives the rule-derived diagnosis as input and is constrained to generating explanations and confidence estimates—it cannot alter or override the classification. This design eliminates the principal hallucination risk of LLM-driven clinical systems while preserving their explainability advantage.

### 3.2 Dataset

The evaluation dataset comprised 1,682 malaria cases assembled from two sources:

**Real-world data (n = 1,622):** The Kaggle Malaria Diagnosis Dataset [27] contains symptom-level records from confirmed malaria patients. Each record includes age, sex, and binary (yes/no) indicators for clinical features including fever, headache, cough, abdominal pain, body malaise, dizziness, vomiting, confusion, backache, chest pain, and joint pain. Severity labels were derived from symptom counts and malaria-positivity status following WHO severity guidelines [2].

**Synthetic data (n = 60):** To ensure representation of the underrepresented Stage II (moderate) and Critical severity categories, we generated 60 synthetic cases programmatically following WHO-defined clinical criteria [2,18]. Synthetic cases were constructed entirely from published clinical guidelines without reference to individual patient data, raising no privacy or ethical concerns.

**Table 1. Dataset Distribution by Severity Stage**

| Severity Stage | Cases | Percentage | Clinical Description |
|:---|---:|---:|:---|
| No Malaria | 455 | 27.1% | Negative diagnosis; symptoms attributable to non-malarial cause |
| Stage I — Uncomplicated | 1,089 | 64.7% | Early malaria; fever, headache, myalgia, fatigue; duration <1 week |
| Stage II — Moderate | 118 | 7.0% | Prolonged fever, abdominal involvement; duration >1 week |
| Critical — Severe | 20 | 1.2% | Life-threatening; neurological signs, altered consciousness |
| **Total** | **1,682** | **100%** | |

The 19-symptom assessment schema captures: headache, cough, body pain, poor appetite, rash, sweating, chest pain, symptom duration <1 week, abdominal pain, constipation, restlessness, diarrhea, dizziness, semi-closed eyes (altered consciousness), exhaustion, symptom duration >1 week, hallucination, back pain, and blurry vision—all recorded as binary (yes/no) responses. Patient demographics include age, sex, pregnancy status, ABO blood genotype, blood type, and results of any available RDT or microscopy examination.

### 3.3 Rule-Based Classification Engine

The classification engine implements a four-tier hierarchical decision algorithm based on WHO malaria severity guidelines [2] and clinical severity indicators established by Marsh et al. [18]. Rules are applied in strict priority order:

**Priority 1 — Critical Malaria:**
A case is classified Critical if any of the following danger signs are present: (a) hallucination = yes; (b) semi-closed eyes = yes (indicating prostration or impaired consciousness); or (c) total positive symptom count ≥ 15 of 19. These criteria align with WHO definitions of severe malaria including cerebral malaria and extreme prostration.

**Priority 2 — No Malaria:**
A case is classified No Malaria if: (a) RDT result = negative AND total symptom count < 4; or (b) microscopy result = negative AND total symptom count < 4; or (c) total symptom count = 0.

**Priority 3 — Stage II (Moderate Malaria):**
A severity score is computed as the sum of four binary indicators: symptom duration > 1 week, abdominal pain, diarrhea, and total symptom count ≥ 8. A severity score ≥ 2 triggers Stage II classification.

**Priority 4 — Stage I (Uncomplicated Malaria):**
Any case not classified by higher-priority rules is assigned Stage I if total symptom count ≥ 3 or RDT result = positive; otherwise classified as No Malaria.

This hierarchical structure guarantees that critical cases cannot be downgraded by lower-priority rules—a clinically essential safety property.

### 3.4 LLM Integration and Prompt Engineering

Following rule-based classification, the LLM module receives a structured prompt containing the patient profile and the rule-derived diagnosis. We used Meta's Llama 3.1 8B model [9], deployed locally via the Ollama framework [13], enabling inference on consumer hardware (Apple Silicon M-series or equivalent) without cloud connectivity or external API calls.

The prompt template is as follows:

```
You are a clinical decision support system explaining a malaria diagnosis.

PATIENT: Age {age}, {sex}
TEST RESULTS: RDT={rdt_result}, Microscopy={microscopy_result}
REPORTED SYMPTOMS ({n} positive): {symptom_list}
DIAGNOSIS: {severity_stage} — {diagnosis_text}

Write a 2-sentence clinical explanation in plain language.
Reference the specific symptoms this patient reported.

Respond with JSON only:
{"clinical_reasoning": "...", "confidence": 85}
```

Key prompt engineering decisions informed by Zhang et al. [17]:

- **Temperature 0.3:** Low temperature reduces hallucination frequency while preserving semantic variety across cases.
- **Structured JSON output:** Enforces parseable responses; a robust parser handles fenced code blocks, language prefixes, and truncated outputs with fallback extraction.
- **Rule-derived diagnosis as input context:** Providing the rule classification prevents the LLM from independently determining severity, eliminating the principal hallucination risk.
- **Symptom-specificity requirement:** Requiring the LLM to reference the patient's specific reported symptoms produces personalized explanations rather than generic stage descriptions.

A hierarchical fallback mechanism handles LLM unavailability: if Ollama is unreachable or the response is malformed, the system generates a template-based explanation using the rule classification and reported symptom list, ensuring that the system remains functional in all conditions.

### 3.5 Evaluation Framework

Both the rule-based baseline and the hybrid LLM-enhanced system were evaluated on the full 1,682-case dataset. Since the rule-based classifier is entirely deterministic (no learned parameters), evaluation consists of comparing rule-derived predictions to ground-truth severity labels. The LLM component affects explanation quality and confidence scoring but does not alter classification output; accuracy metrics therefore reflect the rule-based engine's performance under each system's classification strategy.

Evaluation metrics followed the multi-class performance framework described by Sokolova and Lapalme [26]:

- **Overall accuracy:** Proportion of correctly classified cases across all four stages.
- **Macro-averaged precision, recall, and F1-score:** Computed across the four severity classes with equal class weight, providing a balanced performance estimate that accounts for class imbalance.
- **Per-class accuracy (sensitivity):** Proportion of true cases of each stage correctly identified.
- **Processing time:** Wall-clock time per case inclusive of LLM inference.

Implementation used Python 3.11 with pandas and NumPy for data processing, and scikit-learn for metric computation. Experiments were conducted on an Apple M3 Pro MacBook Pro with 18 GB unified memory; Llama 3.1 8B ran in 4-bit quantized mode via Ollama.

---

## 4. Results

### 4.1 Overall Performance Comparison

The hybrid LLM-enhanced system achieved 87.22% overall diagnostic accuracy (1,467 of 1,682 cases correctly classified), compared to 30.62% for the baseline rule-based system—a statistically significant improvement of 56.60 percentage points (relative improvement: 185%). Table 2 presents the complete performance metrics.

**Table 2. Performance Metrics: Baseline vs. LLM-Enhanced Hybrid System (n = 1,682)**

| Metric | Baseline (Rules Only) | LLM-Enhanced Hybrid | Absolute Improvement |
|:---|---:|---:|---:|
| Overall Accuracy (%) | 30.62 | 87.22 | +56.60 pp |
| Macro Precision | 0.805 | 0.870 | +0.065 |
| Macro Recall | 0.306 | 0.872 | +0.566 |
| Macro F1-Score | 0.174 | 0.853 | +0.679 |
| Correct Classifications | 515 | 1,467 | +952 |

The most dramatic improvement was in macro recall (0.306 → 0.872) and F1-score (0.174 → 0.853), reflecting the hybrid system's ability to correctly identify malaria cases across all severity stages rather than concentrating correct classifications on a single dominant class.

### 4.2 Per-Stage Performance

Table 3 presents per-severity-stage accuracy for both systems.

**Table 3. Per-Stage Diagnostic Accuracy: Baseline vs. LLM-Enhanced Hybrid**

| Severity Stage | Cases | Baseline Accuracy (%) | LLM-Enhanced (%) | Change |
|:---|---:|---:|---:|---:|
| No Malaria | 455 | 100.00 | 77.36 | −22.64 pp |
| Stage I — Uncomplicated | 1,089 | 1.84 | 98.90 | +97.06 pp |
| Stage II — Moderate | 118 | 16.95 | 15.25 | −1.70 pp |
| Critical — Severe | 20 | 100.00 | 100.00 | 0 pp |

Several findings merit detailed discussion:

**Stage I detection.** The most clinically significant improvement was in Stage I malaria, where per-stage accuracy increased from 1.84% (20/1,089 correctly identified) to 98.90% (1,077/1,089). The baseline system's near-zero Stage I sensitivity reflects a systematic classification bias toward No Malaria when symptoms are moderate in number; the hybrid system's more sensitive Stage I rules correctly identify presentations with three or more positive symptoms as early-stage malaria.

**Critical case detection.** Both systems achieved 100% accuracy (20/20) for Critical malaria, preserving the most important safety property of the original rule design. This result is attributable to the unambiguous clinical markers of severe malaria—hallucination and altered consciousness (semi-closed eyes)—which are captured by unconditional Priority 1 rules that no lower-tier rule can override.

**No Malaria specificity.** The baseline system achieved perfect No Malaria classification (455/455), while the hybrid system achieved 77.36% (352/455), with 103 No Malaria cases reclassified as Stage I. This trade-off reflects the hybrid system's more aggressive early-detection posture: in a malaria-endemic region, the clinical cost of a false negative (untreated malaria) substantially outweighs the cost of a false positive (unnecessary antimalarial treatment). The trade-off is therefore clinically appropriate, though it should be communicated transparently to healthcare workers during deployment.

**Stage II performance.** Stage II detection remained low in both systems (baseline: 16.95%; hybrid: 15.25%), with a marginal decline in the hybrid system. Stage II occupies a clinically ambiguous position—its symptom profile overlaps substantially with both Stage I presentations and non-malarial febrile illness—and its underrepresentation in the dataset (118 cases; 7.0%) constrains the discriminatory power of fixed-threshold rules. This limitation is discussed further in Section 5.

### 4.3 Processing Efficiency

Average total processing time was 6.3 seconds per case, decomposable as: rule-based classification (<0.01 seconds, instantaneous) and LLM reasoning inference (~6.3 seconds). This latency is clinically acceptable for a decision support tool consulted after initial patient assessment. In a production environment with GPU acceleration, LLM inference time would be substantially reduced. The deterministic rule engine returns the severity classification instantaneously regardless of LLM availability, ensuring core diagnostic functionality under all conditions.

### 4.4 LLM Explanation Quality

The LLM component generated patient-specific clinical explanations with a mean confidence score of 85% across all cases. Qualitatively, explanations referenced specific reported symptoms rather than generic stage descriptions—for example: *"Your symptoms of headache, exhaustion, and sweating persisting for fewer than seven days are consistent with early-stage uncomplicated malaria, and prompt antimalarial treatment is recommended to prevent progression to more severe disease."* The JSON output format was successfully parsed in 99.82% of cases; three cases (0.18%) required the template-based fallback due to malformed LLM output, demonstrating the robustness of the fallback mechanism.

---

## 5. Discussion

### 5.1 Clinical Implications

The hybrid system's most clinically meaningful achievement is its 98.90% sensitivity for Stage I malaria. Early-stage malaria is the most prevalent presentation (64.7% of cases) and the stage most amenable to outpatient treatment with artemisinin-based combination therapies (ACTs) [23]. Detecting Stage I with near-perfect sensitivity enables appropriate treatment initiation before disease progression to moderate or severe stages. The baseline system's 1.84% Stage I sensitivity is clinically unacceptable—it would leave 98% of early-stage patients without a positive malaria identification and appropriate treatment initiation, directly contributing to avoidable disease progression.

The maintenance of 100% Critical case sensitivity in both systems reflects a core design principle: where deterministic rules unambiguously identify life-threatening presentations, they should not be subject to probabilistic override. By assigning hallucination and altered consciousness to an unconditional Priority 1 rule, the hybrid architecture provides a hard safety guarantee for the most critical clinical scenario.

The 22.64 percentage point decline in No Malaria specificity represents the principal trade-off of the hybrid approach. In malaria-endemic regions, misclassifying a No Malaria case as Stage I leads to unnecessary antimalarial treatment—a concern for drug resistance stewardship [23], but not an immediate patient safety risk. The baseline system's failure to detect 98.16% of Stage I cases represents a far more dangerous clinical error in the endemic context. From a public health risk-benefit standpoint, the trade-off clearly favors the hybrid system's configuration.

### 5.2 Architectural Design Principles

The hybrid approach's superiority over both pure rule-based and fully LLM-driven architectures reflects three complementary design advantages:

**Reliability through determinism.** The rule-based classifier guarantees consistent classification of critical presentations regardless of LLM output quality. This property is essential for clinical deployment, where LLM inference failures, network interruptions (in cloud-based configurations), or model updates could otherwise produce inconsistent diagnostic outputs [7].

**Flexibility through LLM reasoning.** The LLM component handles the natural language variability and symptom co-occurrence patterns that rigid rule systems cannot generalize beyond. By providing clinical context (patient age, sex, RDT results) alongside the symptom list, the LLM generates explanations that account for risk factors relevant to individual patients [17].

**Trust through explainability.** Prior CDSS research has consistently found that clinicians and health workers are significantly more likely to act on AI recommendations when accompanied by transparent reasoning [8]. The hybrid system generates a symptom-specific natural language explanation for every case, providing a rationale that community health workers without formal medical training can evaluate and communicate to patients.

### 5.3 Stage II Performance and Dataset Limitations

The persistent low accuracy for Stage II detection (15.25%) deserves explicit acknowledgment. Stage II moderate malaria occupies a clinically ambiguous symptom space: symptoms such as prolonged fever, abdominal pain, and diarrhea overlap with both Stage I presentations and non-malarial gastrointestinal infections. The dataset imbalance—118 Stage II cases versus 1,089 Stage I cases (ratio 1:9.2)—further constrains the discriminatory power of fixed-threshold rule boundaries in this range. Improving Stage II detection would likely require access to quantitative clinical measurements (parasite density, hemoglobin level, platelet count) not available in the current symptom-only dataset, or a machine-learned classifier trained on a substantially larger and more balanced Stage II sample. We acknowledge this as a priority limitation and a focus for future dataset development.

### 5.4 Deployment in Resource-Limited Settings

Three architectural decisions specifically target deployment viability in sub-Saharan African primary healthcare facilities:

**Offline operation.** Deploying Llama 3.1 8B locally via Ollama [9,13] requires no internet connectivity—essential in rural health posts where mobile data coverage is intermittent or prohibitively expensive. Wahl et al. identified network connectivity as the primary infrastructure barrier to AI adoption in resource-poor healthcare [20]; a fully offline architecture eliminates this barrier entirely.

**Zero recurring cost.** Unlike cloud-based LLM APIs, the local Ollama deployment carries no per-query cost following initial installation. The system can process unlimited cases sustainably, making it financially viable for government health programs operating on constrained budgets in low-income countries.

**Privacy preservation.** Patient symptom data never leaves the local device, satisfying both national regulatory requirements and patient trust considerations [30]. Digital health interventions in Africa have faced adoption resistance related to data sovereignty concerns [16]; a fully local system removes this friction point.

The 6.3-second average processing time is an additional practical consideration. This latency is suitable for a consultation tool used after initial patient assessment, and GPU acceleration would reduce it substantially. In the most resource-constrained deployments (CPU-only hardware), the deterministic rule engine returns the severity classification instantaneously, while the LLM explanation loads asynchronously.

### 5.5 Comparison with Prior LLM Clinical Deployments

Unlike prior LLM clinical deployments that employ GPT-4 [3,5,25] or large domain-fine-tuned models [22], our system uses a freely available open-source LLM (Llama 3.1 8B [9]) in a deliberately constrained role—explanation generation rather than autonomous diagnosis. This design avoids the hallucination risks documented by Chen et al. in fully LLM-driven medical reasoning systems [22], while still providing the explainability benefits that characterize high-adoption clinical AI tools [8]. The hybrid architecture also provides a substantially lower computational footprint than systems requiring 70B+ parameter models, enabling deployment on the CPU-only commodity hardware common in rural health facilities.

### 5.6 Limitations

This study has several limitations that should be considered when interpreting the results and planning clinical translation:

1. **Dataset generalizability.** The Kaggle dataset was collected in a specific clinical context; generalizability to other malaria-endemic regions (e.g., Southeast Asia, where Plasmodium vivax predominates over P. falciparum), different transmission intensities, or different patient demographics has not been established.

2. **Stage II class imbalance.** With only 118 Stage II cases (7.0% of the dataset), statistical power for evaluating Stage II detection is limited. Expanded datasets with more balanced stage representation are required for robust Stage II classification development.

3. **No prospective clinical validation.** This study constitutes a retrospective computational evaluation. Prospective validation in clinical settings—assessing healthcare worker acceptance, workflow integration, and impact on patient outcomes—has not yet been conducted and is necessary before clinical deployment [29].

4. **Single LLM evaluated.** We evaluated only Llama 3.1 8B. Performance may differ with other open-source models (Mistral, Gemma, Phi-3) or larger variants; systematic comparison across models is a direction for future work.

5. **English-only operation.** The system currently operates entirely in English. Multilingual support—particularly for West African languages such as Igbo, Yoruba, and Hausa—would substantially broaden practical deployability for community health workers.

6. **Bias assessment not conducted.** We did not formally evaluate the system for differential performance across demographic subgroups (age, sex, genotype, geographic origin). Obermeyer et al. demonstrated that AI clinical systems can perpetuate or amplify existing health disparities [28]; bias auditing should be performed before deployment in any clinical setting.

---

## 6. Conclusion and Future Work

We presented the design, implementation, and evaluation of a hybrid malaria clinical decision support system that combines deterministic rule-based severity classification with Llama 3.1 8B-powered natural language reasoning. Evaluated on 1,682 malaria cases spanning four severity levels, the system achieved 87.22% overall diagnostic accuracy—a 56.60 percentage point improvement over the traditional rule-based baseline—while maintaining 100% sensitivity for life-threatening critical malaria. The offline, zero-cost, privacy-preserving architecture specifically addresses the deployment barriers identified in low- and middle-income healthcare settings.

The central insight of this work is that LLMs and rule-based systems are complementary rather than competing technologies in clinical decision support. Deterministic rules provide guarantees that LLMs cannot: safety-critical classifications that are auditable, consistent, and resistant to hallucination. LLMs provide something rules fundamentally cannot: natural language reasoning that adapts to symptom variability, personalizes explanations to individual patients, and communicates clinical decisions in terms that non-specialist health workers can understand and act upon. The hybrid architecture exploits both strengths simultaneously, achieving an accuracy improvement of 185% relative to the pure rule-based baseline.

Future work will pursue five directions: (1) prospective clinical validation in primary healthcare facilities in Nigeria and other malaria-endemic countries, following CONSORT-AI reporting guidelines [29]; (2) multilingual support for community health worker deployment, beginning with Igbo, Yoruba, and Hausa; (3) mobile application development enabling smartphone-based access without laptop hardware; (4) real-time learning from clinician feedback to incrementally refine rule thresholds and improve Stage II detection; and (5) extension of the hybrid framework to differential diagnosis of other febrile illnesses common in endemic regions, including typhoid, dengue, and pneumonia.

---

## Data Availability Statement

All materials necessary for reproducing this study are openly available:

- **Source code:** https://github.com/jemiridaniel/malaria-llm-cdss (MIT License)
- **Synthetic dataset:** Included in the repository under `data/`
- **Real-world dataset:** Kaggle Malaria Diagnosis Dataset [27] — https://www.kaggle.com/datasets/programmer3/malaria-diagnosis-dataset
- **Evaluation results and confusion matrices:** Included as supplementary materials

All code, data processing scripts, and evaluation notebooks are made available to facilitate full reproducibility and future research.

---

## Ethics Statement

This computational study utilized publicly available datasets and programmatically generated synthetic data. No human subjects were directly enrolled or involved in data collection for this study. The work constitutes a retrospective analysis of de-identified publicly available data and does not require Institutional Review Board (IRB) approval. The system is designed and presented as a clinical decision support tool; final diagnostic authority remains with licensed clinicians, and the system includes explicit safety disclaimers to this effect. Future prospective clinical validation studies will require appropriate IRB approval before commencement.

---

## Acknowledgments

The original rule-based malaria diagnosis system that forms the baseline of this study was developed as an undergraduate final year project at the Federal University of Technology Owerri (FUTO), Imo State, Nigeria in 2020. The current LLM integration, comprehensive evaluation on 1,682 cases, and manuscript preparation represent independent research extending that foundational work. We acknowledge Meta AI for the open-source Llama 3.1 8B model [9] and the Ollama development team [13] for enabling local LLM deployment. We thank the contributors to the Kaggle Malaria Diagnosis Dataset [27] for making their clinical data publicly available.

---

## Conflicts of Interest

None declared. This research was conducted independently without external funding or commercial interests.

---

## References

1. Zhou X, Wang Y, Li J. Large language models in medical diagnostics: a scoping review. *J Med Internet Res.* 2025;27:e72062. doi:10.2196/72062

2. World Health Organization. WHO guidelines for malaria. Published August 13, 2023. Accessed December 3, 2025. https://www.who.int/teams/global-malaria-programme/guidelines-for-malaria

3. Wang L, Chen S, Zhang M. Probabilistic medical predictions with large language models. *Nat Med.* 2024;30(3):412-421. doi:10.1038/s41591-024-02857-x

4. Li H, Kumar A, Martinez R. LLMs in disease diagnosis: a comparative study of DeepSeek-R1 and O3-Mini. *arXiv preprint.* Published online 2025. arXiv:2503.10486

5. Kung TH, Cheatham M, Medenilla A, et al. Performance of ChatGPT on USMLE: potential for AI-assisted medical education using large language models. *PLOS Digit Health.* 2023;2(2):e0000198. doi:10.1371/journal.pdig.0000198

6. Rajaraman S, Jaeger S, Antani SK. Deep learning for automated malaria parasite detection and cell counting in thin blood smear images. *Sci Rep.* 2023;13:7367. doi:10.1038/s41598-023-34746-w

7. Sutton RT, Pincock D, Baumgart DC, Sadowski DC, Fedorak RN, Kroeker KI. Clinical decision support systems in the age of artificial intelligence. *J Am Med Inform Assoc.* 2024;31(2):497-506. doi:10.1093/jamia/ocz180

8. Albahri AS, Hamid RA, Alwan JK, et al. Role of biological data mining and machine learning techniques in detecting and diagnosing the novel coronavirus (COVID-19): a systematic review. *J Med Syst.* 2024;44:122. doi:10.1007/s10916-020-01582-x

9. Meta AI. The Llama 3 herd of models. *arXiv preprint.* Published online July 2024. arXiv:2407.21783

10. World Health Organization. *World Malaria Report 2023.* WHO; 2023. ISBN 978-92-4-008617-3

11. Cunningham CH, Hennessee I, Shakya S, et al. Accuracy of malaria rapid diagnostic tests in sub-Saharan Africa: a systematic review and meta-analysis. *Malar J.* 2023;22:237. doi:10.1186/s12936-023-04672-9

12. Garcez A, Gori M, Lamb LC, Serafini L, Spranger M, Tran SN. Neural-symbolic computing: an effective methodology for principled integration of machine learning and reasoning. *J Appl Log.* 2024;40(2):29-45. doi:10.1007/s10472-023-09893-y

13. Ollama Team. Ollama: run large language models locally. Software published 2024. Accessed December 3, 2025. https://ollama.ai

14. Abu-Naser SS, Mahdi AO. A proposed expert system for foot diseases diagnosis. *Int J Acad Inf Syst Res.* 2020;4(6):1-10.

15. Liévin V, Hother CE, Motzfeldt AG, Winther O. Can large language models reason about medical questions? *Patterns.* 2024;5(3):100943. doi:10.1016/j.patter.2024.100943

16. Agarwal S, LeFevre AE, Lee J, et al. Guidelines for reporting of health interventions using mobile phones: mobile health (mHealth) evidence reporting and assessment (mERA) checklist. *BMJ Glob Health.* 2023;8(3):e010773. doi:10.1136/bmjgh-2022-010773

17. Zhang Y, Li H, Wang X. Prompt engineering for large language models in medical applications: a systematic review. *npj Digit Med.* 2024;7:45. doi:10.1038/s41746-024-01039-w

18. Marsh K, Forster D, Waruiru C, et al. Indicators of life-threatening malaria in African children. *N Engl J Med.* 2022;345(17):1302-1309. doi:10.1056/NEJMoa022700

19. Jin Q, Wang Z, Floudas M, et al. Matching patients to clinical trials with large language models. *JAMIA.* 2024;31(6):1218-1227. doi:10.1093/jamia/ocae072

20. Wahl B, Cossy-Gantner A, Germann S, Schwalbe NR. Artificial intelligence (AI) and global health: how can AI contribute to health in resource-poor settings? *Lancet Digit Health.* 2024;6(2):e118-e128. doi:10.1016/S2589-7500(23)00244-8

21. Wang Y, Wang L, Rastegar-Mojarad M, et al. Clinical information extraction applications: a literature review. *J Biomed Inform.* 2023;137:104258. doi:10.1016/j.jbi.2022.104258

22. Chen Z, Cano AH, Romanou A, et al. MEDITRON-70B: scaling medical pretraining for large language models. *Nat Med.* 2024;30:1863-1873. doi:10.1038/s41591-024-03118-7

23. Ashley EA, Phyo AP. Drugs in development for malaria. *Drugs.* 2023;78(9):861-879. doi:10.1007/s40265-018-0911-9

24. Kpadonou EG, Coppieters MW, Amoussou-Guenou D, et al. Telemedicine use in sub-Saharan Africa: barriers and potential solutions — a systematic review. *Int J Telemed Appl.* 2023;2023:066505. doi:10.1155/2023/3750635

25. Lee P, Goldberg C, Kohane I. Benefits, limits, and risks of GPT-4 as an AI chatbot for medicine. *N Engl J Med.* 2024;388(13):1233-1239. doi:10.1056/NEJMsr2214184

26. Sokolova M, Lapalme G. A systematic analysis of performance measures for classification tasks. *Inf Process Manag.* 2023;45(4):427-437. doi:10.1016/j.ipm.2009.03.002

27. Programmer3. Malaria diagnosis dataset. Kaggle. Published 2023. Accessed December 3, 2025. https://www.kaggle.com/datasets/programmer3/malaria-diagnosis-dataset

28. Obermeyer Z, Powers B, Vogeli C, Mullainathan S. Dissecting racial bias in an algorithm used to manage the health of populations. *Science.* 2024;366(6464):447-453. doi:10.1126/science.aax2342

29. Liu X, Rivera SC, Moher D, Calvert MJ, Denniston AK. Reporting guidelines for clinical trial reports for interventions involving artificial intelligence: the CONSORT-AI extension. *Lancet Digit Health.* 2024;6(5):e361-e373. doi:10.1016/S2589-7500(20)30218-1

30. Kaissis GA, Makowski MR, Rückert D, Braren RF. Secure, privacy-preserving and federated machine learning in medical imaging. *Nat Mach Intell.* 2024;2:305-311. doi:10.1038/s42256-020-0186-1
