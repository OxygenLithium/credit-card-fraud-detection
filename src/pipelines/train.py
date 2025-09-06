import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, average_precision_score
from utils.metrics import calculate_metrics

def train_model(model, X_train, y_train, cv_config: dict):
    print(f"Training {type(model).__name__}...")
    
    model.fit(X_train, y_train)
    print("Model trained successfully")
    
    # Cross-validation if enabled
    cv_scores = {}
    if cv_config.get('enabled', False):
        cv_scores = perform_cross_validation(model, X_train, y_train, cv_config)
    
    return {
        'model': model,
        'cv_scores': cv_scores,
        'training_samples': len(X_train),
        'feature_count': X_train.shape[1]
    }

def perform_cross_validation(model, X_train, y_train, cv_config):
    folds = cv_config.get('folds', 5)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    scorer = make_scorer(average_precision_score)
    
    print(f"Performing {folds}-fold cross-validation...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scorer)
    
    results = {
        'scores': cv_scores.tolist(),
        'mean': float(cv_scores.mean()),
        'std': float(cv_scores.std()),
        'folds': folds
    }
    
    print(f"CV Average Precision: {results['mean']:.4f} (+/- {results['std']:.4f})")
    
    return results
