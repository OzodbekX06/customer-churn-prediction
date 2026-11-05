import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def evaluate_model(model, X_test, y_test) -> dict:
    predictions = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
        roc_auc = float(roc_auc_score(y_test, probabilities))
    else:
        roc_auc = None

    cm = confusion_matrix(y_test, predictions).tolist()

    return {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "confusion_matrix": cm
    }

def save_metrics_report(metrics_dict: dict, file_path: str):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)
