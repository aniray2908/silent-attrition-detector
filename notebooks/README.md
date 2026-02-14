# Notebooks Overview

This folder contains the modular components of the Silent Attrition Detection system.

Each notebook represents an independent modeling layer within the overall architecture.

---

## 01_hr_attrition_model.ipynb

Structured attrition risk modeling using the IBM HR dataset.

Includes:
- CatBoost classifier
- Class imbalance handling
- Threshold optimization
- SHAP interpretability
- Fairness audit
- Neural tabular model
- Ensemble learning

---

## 02_enron_behavioral_drift.ipynb

Behavioral disengagement detection using internal communication data.

Includes:
- Email header parsing
- Internal employee filtering
- Monthly aggregation
- Rolling baseline modeling
- Drift detection
- Behavioral risk scaling

---

## 03_silent_attrition_fusion.ipynb

Meta-learning fusion layer integrating structured and behavioral risk signals.

Includes:
- Multi-signal simulation
- Nonlinear interaction modeling
- Neural fusion architecture
- Final risk probability generation

---

## System Philosophy

The project follows a modular design:

1. Independent signal modeling
2. Signal standardization
3. Meta-learning fusion

This architecture supports scalability, extensibility, and production readiness.
