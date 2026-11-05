import os
from src.preprocessing import prepare_train_test_data
from src.model import train_logistic_regression, train_knn
from src.evaluation import evaluate_model, save_metrics_report

def run_pipeline():
    dataset_path = os.path.join(os.path.dirname(__file__), "data", "telco_churn.csv")
    X_train, X_test, y_train, y_test, _ = prepare_train_test_data(dataset_path)

    log_reg_model = train_logistic_regression(X_train, y_train)
    knn_model = train_knn(X_train, y_train)

    lr_metrics = evaluate_model(log_reg_model, X_test, y_test)
    knn_metrics = evaluate_model(knn_model, X_test, y_test)

    comparison = {
        "LogisticRegression": lr_metrics,
        "KNeighborsClassifier": knn_metrics
    }

    results_path = os.path.join(os.path.dirname(__file__), "results", "metrics_comparison.json")
    save_metrics_report(comparison, results_path)

    print("Model Evaluation Summary:")
    print("-" * 65)
    print(f"{'Metric':<20} | {'Logistic Regression':<20} | {'KNN':<15}")
    print("-" * 65)
    for metric in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        lr_val = lr_metrics.get(metric)
        knn_val = knn_metrics.get(metric)
        print(f"{metric.capitalize():<20} | {str(lr_val):<20} | {str(knn_val):<15}")
    print("-" * 65)

if __name__ == "__main__":
    run_pipeline()
