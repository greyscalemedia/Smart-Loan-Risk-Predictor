# 🏦 Smart Loan Risk Predictor (ML + Streamlit)

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange?logo=scikitlearn)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-success?logo=xgboost)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## 📌 Project Overview

This project is a **production-ready Smart Loan Risk Predicton system** designed to help financial institutions **assess credit risk** and **predict the probability of loan risk** using machine learning.

The solution combines **advanced ML models** with a **high-end Streamlit dashboard**, delivering an experience similar to real-world fintech products used by banks and NBFCs.

---

## 🎯 Problem Statement

Loan defaults pose significant financial risks for lending institutions.  
The goal of this project is to **predict the likelihood of a borrower defaulting on a loan** using demographic, financial, and loan-specific attributes.

📈 This enables:
- Better credit decisions  
- Early risk identification  
- Data-driven lending strategies  

---

## 🧠 Machine Learning Approach

### ✔ Models Used
- **XGBoost Classifier**
- **Bagging Ensemble Technique**
- Feature importance–based selection
- Class imbalance handling using **sample weighting**

### ✔ Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC (for probability calibration)

---

## 📊 Dataset Description

The dataset contains borrower-level information with the following features:

| Category | Examples |
|--------|---------|
| Demographics | Age, Education, MaritalStatus |
| Financial | Income, CreditScore, DTIRatio |
| Loan Details | LoanAmount, LoanTerm, InterestRate |
| Employment | EmploymentType, MonthsEmployed |
| Target | **Default (0 = No, 1 = Yes)** |

📁 Files:
- `train.csv` → Training data with target  
- `test.csv` → Test data (no target)  
- `prediction_submission.csv` → Final predictions  

---

## 🖥️ Streamlit Web Application

### 🔥 Key Features
- **Modern fintech-style UI**
- Interactive sliders & dropdowns
- Real-time default probability prediction
- Feature importance visualization
- Risk-level interpretation (Low / Medium / High)

### 🎨 UI Highlights
- Glassmorphism cards  
- Gradient theme  
- Interactive Plotly charts  
- Sidebar navigation  

---

## 🛠️ Tech Stack & Tools

### 👨‍💻 Programming & ML
![Python](https://img.shields.io/badge/Python-blue?logo=python)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikitlearn)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?logo=xgboost)

### 📊 Visualization & App
![Matplotlib](https://img.shields.io/badge/Matplotlib-ffffff)
![Seaborn](https://img.shields.io/badge/Seaborn-4EABE6)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?logo=plotly)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit)

---

## 📁 Project Structure
```
Smart-Loan-Risk-Predictor/
│
├── SmartLoanRiskPredictor.ipynb
├── app.py
├── train.csv
├── test.csv
├── prediction_submission.csv
├── requirements.txt
├── README.md
├── .gitignore
└── venv/
```
---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository
```
git clone https://github.com/mr-piyushkr/Smart-Loan-Risk-Predictor.git
cd Smart-Loan-Risk-Predictor
```
---
2️⃣ Create & Activate Virtual Environment
```
python -m venv venv
venv\Scripts\activate   # Windows
```
---
3️⃣ Install Dependencies
```
pip install -r requirements.txt
```
---
4️⃣ Run Streamlit App
```
streamlit run app.py
```
---
🧪 Model Output
- Predicted Probability of default (0–1)
- Risk category:
🟢 Low Risk
🟡 Medium Risk
🔴 High Risk
---
📌 Key Learnings
- End-to-end ML pipeline design
- Handling class imbalance effectively
- Feature engineering & selection
- Ensemble learning with XGBoost
- Deploying ML models using Streamlit
- Designing professional ML dashboards
---

🌐 Future Improvements
- Model monitoring & logging
- API integration (FastAPI)
- Database support
- Cloud deployment (AWS / GCP)
---

## 📄 License

This project is licensed under the MIT License.

---
👨‍💻 Author
Piyush Kumar
