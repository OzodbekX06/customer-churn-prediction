import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_dataset(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

def clean_and_encode_data(df: pd.DataFrame):
    data = df.copy()
    if "customerID" in data.columns:
        data = data.drop(columns=["customerID"])
    
    if "TotalCharges" in data.columns:
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"].astype(str).str.strip(), errors="coerce")
        data["TotalCharges"] = data["TotalCharges"].fillna(data["TotalCharges"].median())
    
    if "Churn" in data.columns:
        data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})
        y = data["Churn"].values
        X = data.drop(columns=["Churn"])
    else:
        y = None
        X = data
    
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dtype=float)
    
    return X, y

def prepare_train_test_data(file_path: str, test_size: float = 0.2, random_state: int = 42):
    raw_df = load_dataset(file_path)
    X, y = clean_and_encode_data(raw_df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = StandardScaler()
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    existing_numeric = [col for col in numeric_cols if col in X_train.columns]
    
    X_train[existing_numeric] = scaler.fit_transform(X_train[existing_numeric])
    X_test[existing_numeric] = scaler.transform(X_test[existing_numeric])
    
    return X_train, X_test, y_train, y_test, scaler
