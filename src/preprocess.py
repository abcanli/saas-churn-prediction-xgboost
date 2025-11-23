import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pickle

CATEGORICAL_COLS = ["country", "plan_type", "acquisition_channel"]
NUMERIC_COLS = [
    "tenure_days",
    "sessions_last_30d",
    "projects_created",
    "team_size",
    "support_tickets_last_90d",
    "nps_score",
    "monthly_spend",
]


def main():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    raw_path = os.path.join(base_dir, "data", "raw", "saas_churn_dataset.csv")
    processed_dir = os.path.join(base_dir, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    df = pd.read_csv(raw_path, parse_dates=["signup_date"])
    df = df[CATEGORICAL_COLS + NUMERIC_COLS + ["is_churned"]].copy()

    X_cat = df[CATEGORICAL_COLS]
    X_num = df[NUMERIC_COLS]
    y = df["is_churned"].astype(int)

    X_train_cat, X_test_cat, X_train_num, X_test_num, y_train, y_test = train_test_split(
        X_cat, X_num, y, test_size=0.2, random_state=42, stratify=y
    )

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_train_cat_enc = ohe.fit_transform(X_train_cat)
    X_test_cat_enc = ohe.transform(X_test_cat)

    scaler = StandardScaler()
    X_train_num_scaled = scaler.fit_transform(X_train_num)
    X_test_num_scaled = scaler.transform(X_test_num)

    X_train = np.hstack([X_train_num_scaled, X_train_cat_enc])
    X_test = np.hstack([X_test_num_scaled, X_test_cat_enc])

    data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train.values,
        "y_test": y_test.values,
        "feature_names_num": NUMERIC_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "ohe": ohe,
        "scaler": scaler,
    }

    out_path = os.path.join(processed_dir, "processed_churn_data.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(data, f)

    print(f"Saved processed data to: {out_path}")
    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)


if __name__ == "__main__":
    main()
