import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from utils.io import create_dir

def run_visualization_pipeline(model, X_test, y_test, feature_names, output_dir):
    create_dir(output_dir)
    
    print("Generating visualizations...")
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        plot_feature_importance(model, feature_names, output_dir)
    
    # Model-specific visualizations
    if hasattr(model, 'predict_proba'):
        plot_prediction_distribution(model, X_test, y_test, output_dir)
    
    print(f"Visualizations saved to {output_dir}")

def plot_feature_importance(model, feature_names, output_dir, top_n = 20):
    if not hasattr(model, 'feature_importances_'):
        print("Model does not have feature_importances_ attribute")
        return
    
    # Get feature importance
    importance = model.feature_importances_
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Plot top features
    plt.figure(figsize=(10, 8))
    top_features = importance_df.head(top_n)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save feature importance data
    importance_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)

def plot_prediction_distribution(model, X_test, y_test, output_dir):
    if not hasattr(model, 'predict_proba'):
        print("Model does not support probability predictions")
        return
    
    # Get prediction probabilities
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Separate by true class
    non_fraud_proba = y_proba[y_test == 0]
    fraud_proba = y_proba[y_test == 1]
    
    # Plot distributions
    plt.figure(figsize=(10, 6))
    plt.hist(non_fraud_proba, bins=50, alpha=0.7, label='Non-Fraud', color='blue', density=True)
    plt.hist(fraud_proba, bins=50, alpha=0.7, label='Fraud', color='red', density=True)
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title('Distribution of Prediction Probabilities')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
