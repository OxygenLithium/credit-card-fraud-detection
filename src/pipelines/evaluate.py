import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    precision_recall_curve, roc_curve, roc_auc_score
)
from utils.metrics import calculate_metrics, find_optimal_threshold
import os
from utils.io import create_dir

def evaluate_model(model, X_train, y_train, X_test, y_test, evaluation_config):
    print("Evaluating model...")
    
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    # Convert to binary
    unique_vals = np.unique(y_pred)
    if not set(unique_vals).issubset({0, 1}):
        if set(unique_vals).issubset({-1, 1}):
            y_pred = np.where(y_pred == -1, 1, 0)
        else:
            print(f"Detected unexpected values {unique_vals}")
    unique_vals = np.unique(y_train_pred)
    if not set(unique_vals).issubset({0, 1}):
        if set(unique_vals).issubset({-1, 1}):
            y_train_pred = np.where(y_train_pred == -1, 1, 0)
        else:
            print(f"Detected unexpected values {unique_vals}")
    
    # Get probabilities if available
    y_proba = None
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, 'decision_function'):
        y_proba = model.decision_function(X_test)
    y_train_proba = None
    if hasattr(model, 'predict_proba'):
        y_train_proba = model.predict_proba(X_train)[:, 1]
    elif hasattr(model, 'decision_function'):
        y_train_proba = model.decision_function(X_train)
    
    metrics = calculate_metrics(y_test, y_pred, y_proba)
    training_metrics = calculate_metrics(y_train, y_train_pred, y_train_proba)
    
    # Find optimal threshold if probabilities available
    threshold_info = None
    if y_proba is not None:
        threshold_config = evaluation_config.get('threshold', {})
        threshold_info = find_optimal_threshold(
            y_test, y_proba,
            metric=threshold_config.get('optimize_for', 'f1'),
            search_space=tuple(threshold_config.get('search_space', [0.01, 0.99])),
            steps=threshold_config.get('steps', 200)
        )
        
        # Recalculate metrics with optimal threshold
        optimal_threshold = threshold_info[0]
        y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
        metrics_optimal = calculate_metrics(y_test, y_pred_optimal, y_proba)
        metrics_optimal['threshold'] = optimal_threshold
    
    # Generate curves
    curves = generate_evaluation_curves(y_test, y_pred, y_proba)
    
    # Print results
    print_evaluation_summary(metrics, threshold_info)
    
    return {
        'metrics': metrics,
        'training_metrics': training_metrics,
        'metrics_optimal': metrics_optimal if threshold_info else None,
        'threshold_info': threshold_info,
        'curves': curves,
        'predictions': {
            'y_pred': y_pred,
            'y_proba': y_proba
        }
    }

def generate_evaluation_curves(y_test, y_pred, y_proba):
    curves = {}
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    curves['confusion_matrix'] = cm.tolist()
    
    # ROC Curve
    if y_proba is not None:
        fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)
        curves['roc_curve'] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': roc_thresholds.tolist(),
            'auc': float(roc_auc)
        }
        
        # Precision-Recall Curve
        precision, recall, pr_thresholds = precision_recall_curve(y_test, y_proba)
        curves['pr_curve'] = {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'thresholds': pr_thresholds.tolist()
        }
    
    return curves

def print_evaluation_summary(metrics, threshold_info):
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    
    if 'average_precision' in metrics:
        print(f"Average Precision: {metrics['average_precision']:.4f}")
    if 'roc_auc' in metrics:
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"True Negatives: {metrics['true_negatives']}")
    print(f"False Positives: {metrics['false_positives']}")
    print(f"False Negatives: {metrics['false_negatives']}")
    print(f"True Positives: {metrics['true_positives']}")
    
    if threshold_info:
        best_threshold, best_score, _ = threshold_info
        print(f"\nOptimal Threshold: {best_threshold:.4f}")
        print(f"Best Score: {best_score:.4f}")
    
    print("="*50)

def plot_evaluation_curves(evaluation_results, output_dir: str):
    create_dir(output_dir)
    curves = evaluation_results['curves']

    if 'roc_curve' in curves:
        plt.figure(figsize=(8, 6))
        roc_data = curves['roc_curve']
        plt.plot(roc_data['fpr'], roc_data['tpr'], 
                label=f'ROC Curve (AUC = {roc_data["auc"]:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    if 'pr_curve' in curves:
        plt.figure(figsize=(8, 6))
        pr_data = curves['pr_curve']
        plt.plot(pr_data['recall'], pr_data['precision'])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'pr_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    if 'confusion_matrix' in curves:
        plt.figure(figsize=(6, 5))
        cm = np.array(curves['confusion_matrix'])
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        classes = ['Non-Fraud', 'Fraud']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Plots saved to {output_dir}")
