[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# Silent Attrition Detector

A multi-signal attrition risk estimation system that integrates structured HR modeling, behavioral communication drift analysis, and neural meta-learning into a unified probabilistic scoring framework.

---

## Overview

Traditional attrition models rely solely on static HR features.  
This project reframes attrition as a **heterogeneous signal fusion problem**, combining:

- Structural organizational risk factors  
- Behavioral disengagement signals  
- A nonlinear neural fusion layer  

The system produces calibrated attrition risk probabilities and functions as a **decision-support risk radar**, not an automated decision engine.

---

## System Architecture

Structured HR Model (CatBoost + Neural Tabular)\
↓\
Behavioral Drift Engine (Email Communication Analysis)\
↓\
Neural Meta-Fusion Layer\
↓\
Final Attrition Risk Score

### HR Modeling
- CatBoost with native categorical handling  
- Neural tabular model (PyTorch embeddings)  
- Class imbalance handling  
- Threshold optimization  
- SHAP explainability  
- Fairness audit (Age sensitivity testing)

### Behavioral Drift Engine
- Internal email filtering  
- Rolling 6-month communication baseline  
- Drift calculation:

  Drift = (Current − Rolling Avg) / Rolling Avg  

- Z-score normalization and risk scaling

### Meta-Fusion Layer
- Combines HR and behavioral risk signals  
- Nonlinear interaction modeling  
- Dense neural architecture  
- Calibrated probability output

---

## Model Performance

| Model | ROC-AUC |
|-------|---------|
| CatBoost | 0.81 |
| Neural Tabular | 0.78 |
| Ensemble | 0.819 |

The ensemble model improves robustness by capturing complementary structural and behavioral signals.

---

### ROC Curve — Ensemble Model

![ROC Curve](results/hr_roc_curve.png)

---

### SHAP Feature Importance

![SHAP Summary](results/shap_summary.png)

Key drivers include:
- Overtime  
- Job Role  
- Stock Option Level  

---

## Example Output

Employee 1023 → Attrition Risk: 87%\
Employee 2451 → Attrition Risk: 22%\
Employee 3817 → Attrition Risk: 64%

Risk amplification occurs when structural stress aligns with measurable behavioral disengagement.

---

## Key Insights

- Overtime strongly increases attrition probability.  
- Stock options reduce attrition risk.  
- Behavioral communication drift acts as an early disengagement signal.  
- Multi-signal fusion captures risk interactions better than isolated models.

---

## Setup & Reproduction

1. Clone the repository:

git clone https://github.com/aniray2908/silent-attrition-detector.git
cd silent-attrition-detector

2. Install dependencies:

pip install -r requirements.txt

3. Download datasets and place inside `data/`:

IBM HR Dataset  
https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset  

Enron Email Dataset  
https://www.kaggle.com/datasets/wcukierski/enron-email-dataset  

4. Run notebooks in order:

01_hr_model.ipynb
02_behavioral_drift.ipynb
03_meta_fusion.ipynb

---

## Ethical Considerations

- Age feature audited and removed after fairness sensitivity testing.  
- Designed as decision-support, not automated employment decision system.  
- Focused on structural and behavioral signals rather than protected demographic attributes.

---

## Future Extensions

- Communication network centrality metrics  
- Email sentiment modeling  
- Real-world joint HR + behavioral datasets  
- Causal uplift modeling  
- API deployment for real-time scoring  

---

## Author

Anisha Ray  

---

## License

This project is licensed under the MIT License.

