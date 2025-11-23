# SaaS Churn Prediction – XGBoost & Streamlit Dashboard

A end-to-end **customer churn prediction** project for a subscription-based SaaS product.  
The project combines:

- ✅ **Synthetic SaaS customer dataset**
- ✅ **Feature engineering & preprocessing pipeline**
- ✅ **XGBoost churn classifier**
- ✅ **Interactive Streamlit app** for:
  - Individual churn risk scoring
  - Scenario testing (what-if analysis)
  - Model performance overview

Bu repo; Data / ML / Analytics / Product Analyst başvurularında portföy projesi olarak kullanılabilecek şekilde tasarlandı.

---

## 🚀 Highlights

- **Binary churn prediction** (`churned` vs `active`)
- Features include:
  - Product usage (logins, feature usage, last_seen, etc.)
  - Billing & subscription signals (plan type, MRR, discounts)
  - Customer profile (country, segment, company size…)
- **XGBoost** model with:
  - Class balancing
  - Train / validation / test split
  - Metrics: accuracy, precision, recall, F1, ROC-AUC
- **Streamlit app**:
  - Sidebar form ile müşteri profili gir
  - Anında **“Low / Medium / High churn risk”** skoru
  - Model metrikleri & confusion matrix görselleri
- Reproducible pipeline (scripts under `src/`)

---

## 📸 Screenshots

![High Risk](assets/assetsscreenshot_high_risk.png)


![Low Risk](assets/assetsscreenshot_low_risk.png)

![Metrics](assets/assetsscreenshot_metrics.png)

---

## 🧱 Project Structure

```bash
saas-churn-prediction-xgboost/
│
├── README.md
├── requirements.txt
├── app.py                     # Streamlit churn risk dashboard
├── assets/                    # UI & metrics screenshots
│   ├── assetsscreenshot_high_risk.png
│   ├── assetsscreenshot_low_risk.png
│   └── assetsscreenshot_metrics.png
│
├── data/
│   ├── raw/
│   │   └── churn_customers_raw.csv      # Synthetic raw dataset
│   └── processed/
│       └── churn_processed.parquet      # Preprocessed dataset
│
├── outputs/
│   ├── models/
│   │   └── xgboost_churn_model.json     # Trained model
│   └── metrics/
│       └── classification_report.txt
│
└── src/
    ├── config.py              # Paths, feature lists, constants
    ├── generate_synthetic_data.py   # Synthetic SaaS churn dataset
    ├── preprocess.py          # Cleaning, encoding, train/val/test split
    ├── train_xgboost.py       # Model training + evaluation
    └── utils.py               # Helper functions (logging, metrics etc.)

⚙️ Installation
git clone https://github.com/abcanli/saas-churn-prediction-xgboost.git
cd saas-churn-prediction-xgboost

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
# source venv/bin/activate

pip install -r requirements.txt

🧪 Data & Preprocessing
1️⃣ (optional)
python src/generate_synthetic_data.py

2️⃣ Preprocess Pipeline
python src/preprocess.py

🤖 Model Training – XGBoost
python src/train_xgboost.py

📊 Streamlit App – Churn Risk Dashboard
streamlit run app.py
🧠 Tech Stack

Python

Pandas, NumPy

scikit-learn

XGBoost

Streamlit – interactive dashboard

👤 Author

Ali Berk Canlı
NLP/ML Analyst • Data / Product Analytics

GitHub: https://github.com/abcanli
LinkedIn: https://www.linkedin.com/in/aliberkcanlı
