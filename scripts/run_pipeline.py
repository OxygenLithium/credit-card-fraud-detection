#!/usr/bin/env python3

import argparse
import yaml
import sys
import os
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from pipelines.data import prepare_data, get_data_summary
from pipelines.train import train_model
from pipelines.evaluate import evaluate_model, plot_evaluation_curves
from pipelines.visualize import run_visualization_pipeline
from pipelines.factory import build_model
from utils.io import save_model, save_json, create_dir
from utils.seed import set_global_seed


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def run_pipeline(config_path):
    print("Pipeline start\n")
    
    config = load_config(config_path)
    print(f"Configuration loaded from: {config_path}")
    
    set_global_seed(config['seed'])
    
    # Get datetime to add timestamp to folders
    current_datetime = datetime.now()
    time_string = current_datetime.isoformat()

    output_dir = config['outputs']['dir']
    output_dir += "_" + time_string
    create_dir(output_dir)
    print(f"Output directory: {output_dir}")
    
    print("\nDATA PREPARATION")
    X_train, X_test, y_train, y_test, feature_names, scaler = prepare_data(config)
    
    data_summary = get_data_summary(
        X_train.join(y_train), 
        config['data']['target']
    )
    print(f"Data summary: {data_summary}")
    
    print("\nMODEL BUILDING")
    model = build_model(config['model']['name'], config['model']['params'])
    
    print("\nMODEL TRAINING")
    training_results = train_model(model, X_train, y_train, config['evaluation']['cv'])
    
    print("\nMODEL EVALUATION:")
    evaluation_results = evaluate_model(model, X_test, y_test, config['evaluation'])
    
    if config['outputs'].get('save_plots', True):
        print("\nGENERATING VISUALIZATIONS")
        run_visualization_pipeline(model, X_test, y_test, feature_names, output_dir)
    
    print("\nSAVING RESULTS")
    
    if config['outputs'].get('save_model', True):
        model_path = os.path.join(output_dir, 'model.pkl')
        save_model(model, model_path)
    
    if scaler is not None:
        scaler_path = os.path.join(output_dir, 'scaler.pkl')
        save_model(scaler, scaler_path)
    
    # Save metrics
    if config['outputs'].get('save_metrics', True):
        metrics_data = {
            'config': config,
            'data_summary': data_summary,
            'training_results': {
                'cv_scores': training_results['cv_scores'],
                'training_samples': training_results['training_samples'],
                'feature_count': training_results['feature_count']
            },
            'evaluation_results': {
                'metrics': evaluation_results['metrics'],
                'metrics_optimal': evaluation_results['metrics_optimal'],
                'threshold_info': evaluation_results['threshold_info']
            }
        }
        
        metrics_path = os.path.join(output_dir, 'metrics.json')
        save_json(metrics_data, metrics_path)
    
    # Save evaluation plots
    if config['outputs'].get('save_plots', True):
        plot_evaluation_curves(evaluation_results, output_dir)
    
    print("\n" + "="*60)
    print("Pipeline finished")
    print("="*60)
    print(f"Results saved to {output_dir}")
    
    return {
        'model': model,
        'training_results': training_results,
        'evaluation_results': evaluation_results,
        'output_dir': output_dir
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--output-dir', help='Override output directory. Unsure if this currently works')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    
    try:
        results = run_pipeline(args.config)
        
        # if args.output_dir:
        #     print(f"Note: Output directory overridden to: {args.output_dir}")
        
    except Exception as e:
        print(f"Error running pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
