---
🧠 FULL PROFESSIONAL README.md (tam sürüm – direkt yapıştır)
# 🔮 SaaS Churn Prediction Dashboard  
**Real-Time Churn Risk Scoring • ML Models • SHAP Explainability • Streamlit UI**

This project builds a complete, production-style **SaaS customer churn prediction system**, including:

- A **synthetic SaaS customer dataset**
- **Feature preprocessing pipeline**  
- Machine learning models (Logistic Regression + XGBoost)
- **SHAP explainability**
- Interactive **Streamlit dashboard** for real-time churn scoring

Perfect for:
✔ Product Teams  
✔ Customer Success  
✔ SaaS Founders  
✔ Data Science / ML Engineer portfolios  
✔ Interview case studies  

---

# ⭐ Key Features

### 🔥 Machine Learning  
- Logistic Regression baseline  
- XGBoost high-performance classifier  
- Full evaluation reports  
- Confusion matrix + classification report  

---

# 🧹 Data Processing  
- Categorical encoding  
- Numerical feature scaling  
- Train / validation / test split  
- Automatic dataset sanity checks  
- Synthetic data generator included  

---

# 🧾 Model Explainability (SHAP)  
- SHAP summary plot  
- Feature importance  
- Per-customer feature contribution  
- Why the model predicted *high risk* vs *low risk*

---

# 🎛 Streamlit Dashboard (Interactive)

The dashboard allows you to:

✔ Input a customer profile  
✔ Get instant *churn probability*  
✔ View model explanation  
✔ See which features push risk up/down  
✔ Visualize retention metrics  

---

# 📸 Screenshots

### 🔵 High Churn Risk Example
<img src="assets/screenshot_high_risk.png" width="650">

---

### 🟢 Low Churn Risk Example
<img src="assets/screenshot_low_risk.png" width="650">

---

### 📊 Model Evaluation & Confusion Matrix
<img src="assets/screenshot_metrics.png" width="650">

---

# 📂 Project Structure



SaaSChurn/
│
├── data/
│ ├── raw/
│ └── processed/
│
├── src/
│ ├── preprocess.py
│ ├── train_model.py
│ ├── predict.py
│ ├── explain.py
│ └── utils.py
│
├── models/
│ ├── logistic_regression.pkl
│ ├── xgboost_model.json
│
├── assets/
│ ├── screenshot_high_risk.png
│ ├── screenshot_low_risk.png
│ └── screenshot_metrics.png
│
├── app.py
├── requirements.txt
└── README.md

# 🚀 How to Run the Project

### **1. Create virtual environment**
```bash
python -m venv venv
2. Activate
Windows

bash
Kodu kopyala
venv\Scripts\activate
Mac/Linux

bash
Kodu kopyala
source venv/bin/activate
3. Install dependencies
bash
Kodu kopyala
pip install -r requirements.txt
🧪 Run Preprocessing
bash
Kodu kopyala
python src/preprocess.py
🤖 Train the ML Models
bash
Kodu kopyala
python src/train_model.py
This trains:

Logistic Regression

XGBoost

and saves them into models/.

🌐 Launch Streamlit App
bash
Kodu kopyala
streamlit run app.py
The dashboard opens at:

👉 http://localhost:8501/

📊 Example Model Performance
Model	Accuracy	F1-Score	Notes
Logistic Regression	~0.85	~0.84	Strong baseline
XGBoost	~0.92	~0.91	Best performer
SHAP	—	Explainability	Per-customer reasoning

🔍 Why This Project is Strong for Your Portfolio?
End-to-end pipeline

Realistic SaaS dataset & business context

Multiple models + comparison

Explainability (SHAP) → interview gold

Interactive dashboard

Clean architecture

This is the type of project that hiring managers love because it shows both
ML engineering + analytics/product thinking.

👤 Author
Ali Berk Canlı
Data Scientist • ML Engineer • SaaS Analytics
GitHub: https://github.com/abcanli
LinkedIn: https://www.linkedin.com/in/aliberkcanlı

