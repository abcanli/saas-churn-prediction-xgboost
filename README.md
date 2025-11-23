# SaaS Churn Prediction – XGBoost & Streamlit

End-to-end churn prediction pipeline on a synthetic SaaS customer dataset.

## Steps

1. Generate data
```bash
python src/generate_dataset.py
```

2. Preprocess
```bash
python src/preprocess.py
```

3. Train model
```bash
python src/train_model.py
```

4. Run app
```bash
streamlit run app.py
```
