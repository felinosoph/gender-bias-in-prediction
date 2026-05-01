# Gender Bias in Occupation Prediction: A BiasBios Analysis

## Project Description

The inspection motivates a controlled debiasing intervention: a second model 
is trained with a targeted pronoun stop list, holding all other hyperparameters 
constant. The two models are compared on both classification performance and 
gender signal strength, demonstrating that interpretability can enable precise, 
measurable bias reduction.

---

## Dataset

BiasBios (De-Arteaga et al., 2019)
Microsoft BiosBias GitHub: [https://github.com/Microsoft/biosbias](https://github.com/Microsoft/biosbias)
HuggingFace BiosBias Dataset: [https://huggingface.co/datasets/LabHC/bias_in_bios](https://huggingface.co/datasets/LabHC/bias_in_bios)

---

## Approach

- Features: TF-IDF on biography text
- Main model: Logistic regression (multiclass occupation prediction)
- Evaluation: Accuracy, macro F1, per-class F1
- Ethics analysis: Top LogReg weights per occupation class, logit contributions correlated against ground truth gender labels to surface gendered vocabulary driving predictions
- Debiasing intervention: second model trained with targeted pronoun stop list 
  (she, her, hers, he, his, him, ms, mr, mrs), all other hyperparameters held constant
- Intervention evaluation: macro F1 and mean |r| compared across both models

---

## Repository Structure

```
.
├── README.md
├── modeling.ipynb
├── Machine_Learning_Analysis_Report.pdf
├── biasbios.parquet
├── requirements.txt
└── report/
    ├── figures/
    │   ├── gender_bias_correlation_0.png
    │   ├── gender_bias_correlation_1.png
    |   └── gender_bias_comparison.png
    ├── module_summary.tex
    └── references.bib
```

---

## How to Run

```bash
git clone https://github.com/felinosoph/gender-bias-in-prediction
cd gender-bias-in-prediction
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

Then open `modeling.ipynb` and select Restart Kernel and Run All Cells.

---

## Bias and Data Quality

BiasBios was constructed from Common Crawl biography pages and may reflect structural biases in who receives online biographical coverage. Occupational categories are not uniformly distributed across gender, which affects both training data balance and evaluation. The ethics analysis extracts the top TF-IDF features by LogReg weight per occupation class and computes the point-biserial correlation of their logit contributions with the ground truth gender label. This surfaces gendered vocabulary that the occupation classifier relies on, even though gender is not an input feature. The analysis is mechanistic and diagnostic: it locates where bias lives in the model rather than auditing fairness end-to-end. Logistic regression makes this analysis possible because the weight matrix is directly interpretable. Recovering this kind of interpretability as model capacity grows is the central question the subsequent projects in this capstone explore.

---

## Future Integration Reflections

**ML Workflow Changes:** To extend this pipeline, the TF-IDF representation could be replaced with contextual embeddings (e.g. BERT CLS tokens) to capture richer semantic content. The weight-based bias analysis could be generalised to a fairness monitoring layer at inference time, flagging predictions where gender-correlated features dominate the decision. Cross-validation across occupational subgroups would strengthen robustness claims.

**Neural Network Preparation:** Biography text would be tokenized and passed through a pretrained encoder. The occupation classifier would become a classification head trained on frozen or fine-tuned representations. Weight inspection is no longer directly available at this scale, so recovering interpretability requires active methods: attention analysis, probing classifiers, or sparse autoencoders. Class imbalance across occupations would require loss weighting or stratified sampling.

**Agentic Automation:** An automated hiring or screening agent could integrate this pipeline as a bias monitoring tool. Before returning an occupation prediction, the agent would compute the gender correlation of the top active features and surface a bias risk score to a human reviewer if it exceeds a threshold. This creates a human-in-the-loop checkpoint grounded in the model's internals, rather than applying blanket review to all predictions.

---

## Capstone Arc

This project is part of a seven-project MSc capstone exploring how bias surfaces, propagates, and can be probed across NLP and ML systems.

| Project | Dataset  | Focus                                              |
|---------|----------|----------------------------------------------------|
| P1      | COMPAS   | Bias in criminal risk scoring                      |
| P2      | Jigsaw   | Statistical patterns in toxicity data              |
| P3      | BiasBios | Gender bias in occupation prediction               |
| P4 - P7 | TBD      | Escalating model complexity and interpretability depth |

Projects build progressively toward a final synthesis of bias measurement across the arc.