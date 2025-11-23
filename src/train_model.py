import os
import pickle
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt


def main():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    processed_path = os.path.join(base_dir, "data", "processed", "processed_churn_data.pkl")
    outputs_dir = os.path.join(base_dir, "outputs")
    plots_dir = os.path.join(outputs_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    with open(processed_path, "rb") as f:
        data = pickle.load(f)

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    print("Training XGBoost model...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print("===== TEST METRICS =====")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-churn", "Churn"])
    ax.set_yticklabels(["Non-churn", "Churn"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.colorbar(im)
    fig.tight_layout()
    cm_path = os.path.join(plots_dir, "confusion_matrix.png")
    fig.savefig(cm_path)
    print(f"Saved confusion matrix to: {cm_path}")

    from sklearn.metrics import RocCurveDisplay
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax2)
    ax2.set_title("ROC Curve - Churn Model")
    fig2.tight_layout()
    roc_path = os.path.join(plots_dir, "roc_curve.png")
    fig2.savefig(roc_path)
    print(f"Saved ROC curve to: {roc_path}")

    fi = model.feature_importances_
    fi_path = os.path.join(outputs_dir, "feature_importances.npy")
    np.save(fi_path, fi)

    model_path = os.path.join(outputs_dir, "xgb_churn_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()
