import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def simulate_users(n_users: int = 8000) -> pd.DataFrame:
    countries = ["DE", "IT", "CH", "TR", "US", "FR", "NL", "SE"]
    plans = ["free", "pro", "enterprise"]
    channels = ["ads", "organic", "referral", "partner"]

    today = datetime(2025, 1, 1)
    signup_start = today - timedelta(days=365)

    user_ids = [f"U{idx:05d}" for idx in range(1, n_users + 1)]
    signup_dates = [
        signup_start + timedelta(days=int(np.random.randint(0, 365))) for _ in range(n_users)
    ]

    df = pd.DataFrame(
        {
            "user_id": user_ids,
            "country": np.random.choice(countries, size=n_users, p=[0.2, 0.15, 0.1, 0.15, 0.15, 0.1, 0.1, 0.05]),
            "plan_type": np.random.choice(plans, size=n_users, p=[0.6, 0.3, 0.1]),
            "acquisition_channel": np.random.choice(channels, size=n_users, p=[0.35, 0.4, 0.2, 0.05]),
            "signup_date": signup_dates,
        }
    )

    df["tenure_days"] = (today - df["signup_date"]).dt.days.clip(lower=1)
    base_sessions = np.random.gamma(shape=2.0, scale=3.0, size=n_users)
    plan_multiplier = df["plan_type"].map({"free": 0.7, "pro": 1.2, "enterprise": 1.5})
    df["sessions_last_30d"] = (base_sessions * plan_multiplier * np.random.uniform(0.6, 1.4, size=n_users)).round().astype(int)
    df["sessions_last_30d"] = df["sessions_last_30d"].clip(lower=0)

    df["projects_created"] = np.random.poisson(lam=2.0, size=n_users)
    df["team_size"] = np.random.choice([1, 2, 3, 4, 5, 10, 20], size=n_users, p=[0.15, 0.2, 0.2, 0.15, 0.1, 0.1, 0.1])

    df["support_tickets_last_90d"] = np.random.poisson(lam=0.8, size=n_users)

    base_nps = df["plan_type"].map({"free": 15, "pro": 30, "enterprise": 40}).astype(float)
    noise = np.random.normal(0, 15, size=n_users)
    df["nps_score"] = (base_nps + noise).clip(-100, 100).round().astype(int)

    df["monthly_spend"] = df["plan_type"].map({"free": 0.0, "pro": 49.0, "enterprise": 199.0}).astype(float)
    df["monthly_spend"] *= np.random.uniform(0.7, 1.3, size=n_users)
    df["monthly_spend"] = df["monthly_spend"].round(2)

    norm_sessions = (df["sessions_last_30d"] / (df["sessions_last_30d"].max() + 1e-6))
    norm_nps = (df["nps_score"] + 100) / 200.0
    norm_tenure = df["tenure_days"] / (df["tenure_days"].max() + 1e-6)
    norm_tickets = df["support_tickets_last_90d"] / (df["support_tickets_last_90d"].max() + 1e-6)
    norm_spend = df["monthly_spend"] / (df["monthly_spend"].max() + 1e-6)

    score = (
        -1.5 * norm_sessions
        - 1.2 * norm_nps
        - 0.8 * norm_tenure
        + 1.0 * norm_tickets
        - 0.5 * norm_spend
    )

    score += df["plan_type"].map({"free": 0.3, "pro": 0.0, "enterprise": -0.2}).values

    churn_prob = 1 / (1 + np.exp(-score))
    df["churn_probability"] = churn_prob

    df["is_churned"] = np.random.binomial(1, churn_prob).astype(int)

    return df


def main():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    raw_dir = os.path.join(base_dir, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    df = simulate_users(n_users=8000)
    out_path = os.path.join(raw_dir, "saas_churn_dataset.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved synthetic churn dataset to: {out_path}")
    print(df.head())
    print(df['is_churned'].value_counts(normalize=True))


if __name__ == "__main__":
    main()
