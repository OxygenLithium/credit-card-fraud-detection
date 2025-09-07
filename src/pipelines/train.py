import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, average_precision_score
from pipelines.factory import build_model

def train_model(model, X_train, y_train, cv_config, modelName, modelParams):
    print(f"Training {type(model).__name__}...")
    
    model.fit(X_train, y_train)
    print("Model trained successfully")
    # Put in training accuracy
    
    # Cross-validation if enabled
    cv_scores = {}
    if cv_config.get('enabled', False):
        cv_scores = perform_cross_validation(X_train, y_train, cv_config, modelName, modelParams)
    
    return {
        'model': model,
        'cv_scores': cv_scores,
        'training_samples': len(X_train),
        'feature_count': X_train.shape[1]
    }

def perform_cross_validation(X_train, y_train, cv_config, modelName, modelParams):
    folds = cv_config.get('folds', 5)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    scorer = make_scorer(average_precision_score)

    model = build_model(modelName, modelParams)
    
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
