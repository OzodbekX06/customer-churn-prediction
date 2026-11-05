import unittest
import pandas as pd
import numpy as np
from src.preprocessing import clean_and_encode_data
from src.model import train_logistic_regression, train_knn
from src.evaluation import evaluate_model

class TestChurnPipeline(unittest.TestCase):
    def setUp(self):
        self.sample_df = pd.DataFrame({
            "customerID": ["C1", "C2", "C3", "C4", "C5", "C6"],
            "gender": ["Male", "Female", "Male", "Female", "Male", "Female"],
            "tenure": [1, 12, 24, 5, 48, 60],
            "MonthlyCharges": [29.85, 56.95, 53.85, 42.30, 80.0, 95.5],
            "TotalCharges": ["29.85", "683.4", "1292.4", " ", "3840.0", "5730.0"],
            "Churn": ["No", "No", "Yes", "No", "Yes", "No"]
        })

    def test_clean_and_encode_data(self):
        X, y = clean_and_encode_data(self.sample_df)
        self.assertNotIn("customerID", X.columns)
        self.assertEqual(len(y), 6)
        self.assertFalse(np.isnan(X["TotalCharges"]).any())

    def test_model_training_and_eval(self):
        X, y = clean_and_encode_data(self.sample_df)
        model = train_logistic_regression(X, y)
        metrics = evaluate_model(model, X, y)
        self.assertIn("accuracy", metrics)
        self.assertIn("f1_score", metrics)
        self.assertGreaterEqual(metrics["accuracy"], 0.0)

if __name__ == "__main__":
    unittest.main()
