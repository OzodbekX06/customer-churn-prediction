from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

def train_logistic_regression(X_train, y_train, C: float = 1.0, max_iter: int = 1000):
    model = LogisticRegression(C=C, max_iter=max_iter, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_knn(X_train, y_train, n_neighbors: int = 5):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model
