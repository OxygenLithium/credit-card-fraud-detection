from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def build_model(name: str, params: dict):
    # Build model based on name and params
    # Name must be found in this map:
    model_map = {
        'random_forest': RandomForestClassifier,
        'isolation_forest': IsolationForest,
        'logistic_regression': LogisticRegression,
        'svm': SVC,
        'knn': KNeighborsClassifier,
    }
    
    if name not in model_map:
        raise ValueError(f"Unknown model: {name}. Available models: {list(model_map.keys())}")
    
    model_class = model_map[name]
    return model_class(**params)
