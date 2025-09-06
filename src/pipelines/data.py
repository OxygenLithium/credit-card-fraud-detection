# Prepares data for pipeline and does splitting
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

def load_data(config):
    data_path = config['data']['path']
    df = pd.read_csv(data_path)
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def prepare_data(config):
    df = load_data(config)
    
    # Separate features and target
    target = config['data']['target']
    X = df.drop(target, axis=1)
    y = df[target]
    feature_names = X.columns.tolist()
    
    print(f"Features: {len(feature_names)}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Train-test split
    test_size = config['data']['test_size']
    stratify = config['data'].get('stratify', True)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=config['seed'],
        stratify=y if stratify else None
    )
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Data is scaled and transformed here
    scaler = None
    if config['data'].get('scale', True):
        scaler = StandardScaler()
        X_train = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=feature_names,
            index=X_train.index
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test),
            columns=feature_names,
            index=X_test.index
        )
        print("Data scaled")
    
    # Balancing
    balance_method = config['data'].get('balance', 'none')
    sampling_strategy = config['data'].get('sampling_strategy', 1)
    
    if balance_method != 'none':
        print(f"Balancing data using {balance_method}...")
        X_train, y_train = balance(X_train, y_train, balance_method, sampling_strategy)
        print(f"Data balanced using {balance_method}")
        print(f"Balanced train set: {X_train.shape[0]} samples")
        print(f"Balanced target distribution: {y_train.value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test, feature_names, scaler


def balance(X_train, y_train, method, sampling_strategy = 1):
    if method == 'smote':
        sampler = SMOTE(random_state=42, sampling_strategy=sampling_strategy)
    elif method == 'undersample':
        sampler = RandomUnderSampler(random_state=42, sampling_strategy=sampling_strategy)
    elif method == 'smote_tomek':
        sampler = SMOTETomek(random_state=42, sampling_strategy=sampling_strategy)
    else:
        raise ValueError(f"Unknown balancing method: {method}")
    
    X_balanced, y_balanced = sampler.fit_resample(X_train, y_train)
    
    if isinstance(X_train, pd.DataFrame):
        X_balanced = pd.DataFrame(X_balanced, columns=X_train.columns)
    
    return X_balanced, y_balanced

def get_data_summary(df, target):
    summary = {
        'total_samples': len(df),
        'total_features': len(df.columns) - 1,
        'target_distribution': df[target].value_counts().to_dict(),
        'fraud_rate': df[target].mean(),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
    }
    
    return summary
