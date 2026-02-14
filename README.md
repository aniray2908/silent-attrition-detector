# Silent Attrition Detector

A modular, multi-signal machine learning system designed to detect employee attrition risk using both structured HR data and behavioral communication drift signals.

This project demonstrates production-style ML system design, moving beyond simple classification toward layered, explainable risk modeling.

---

## 🔎 Project Motivation

Traditional attrition models rely solely on structured HR features such as job role, overtime, and compensation.

However, employee disengagement often manifests behaviorally before formal resignation.

This system integrates:

- Structured HR attrition risk modeling
- Behavioral communication drift detection
- Neural meta-learning fusion

to create a unified **Silent Attrition Risk Score**.

---

## 🏗 System Architecture

Structured HR Model (CatBoost + Neural Tabular)\
↓\
Behavioral Drift Engine (Email Communication Analysis)\
↓\
Neural Meta-Fusion Layer\
↓\
Final Attrition Risk Score\

Each module is developed independently and integrated via a meta-learning layer.

---

## 📂 Repository Structure
silent-attrition-detector/\
│\
├── notebooks/\
│ ├── 01_hr_attrition_model.ipynb\
│ ├── 02_enron_behavioral_drift.ipynb\
│ └── 03_silent_attrition_fusion.ipynb\
│\
├── requirements.txt\
└── README.md\


---

## 📘 Notebook Descriptions

### 01_hr_attrition_model.ipynb
- IBM HR dataset modeling
- CatBoost classifier
- Class imbalance handling
- Threshold optimization
- SHAP interpretability
- Fairness audit (Age sensitivity analysis)
- Neural tabular modeling (PyTorch)
- Ensemble learning

### 02_enron_behavioral_drift.ipynb
- Raw email parsing
- Internal communication filtering
- Monthly aggregation
- Rolling baseline computation
- Drift detection
- Behavioral risk standardization (0–1 scale)

### 03_silent_attrition_fusion.ipynb
- Multi-signal simulation
- Nonlinear interaction modeling
- Neural meta-learning fusion network
- Risk probability calibration
- System-level integration logic

---

## 🧠 Key Technical Highlights

- Imbalanced classification optimization
- Probability threshold calibration
- SHAP-based model explainability
- Responsible AI through feature sensitivity auditing
- Neural tabular architecture with embeddings
- Ensemble stacking
- Multi-modal signal fusion
- Modular system design

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- CatBoost
- SHAP
- PyTorch
- Matplotlib

---

## 📈 Why This Project Stands Out

Unlike isolated ML notebooks, this project demonstrates:

- Architectural thinking
- Multi-signal risk modeling
- System modularity
- Ethical model considerations
- Industry-aligned meta-learning design

It mirrors how modern churn, fraud, and credit risk systems are built.

---

## 🚀 Future Extensions

- Network centrality modeling for behavioral analysis
- Sentiment analysis integration
- Real-world dataset alignment
- REST API deployment
- Real-time scoring pipeline

---

## Author

Anisha Ray

