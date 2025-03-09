import os
import random
import joblib
import numpy as np
from skopt import BayesSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer, StandardScaler
from model import choose_base_model
from helpers.feature_preparation import DataProcessor
from helpers.utils import (
    calculate_metrics,
    save_best_model_predictions, 
    select_features_with_high_correlation)
  

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def train_model(data_dict, selected_features, results_output_path, args): 
    
    kbins = KBinsDiscretizer(n_bins=args.n_bins, encode='ordinal', strategy='quantile')
    y_binned = kbins.fit_transform(data_dict['yield'].values.reshape(-1, 1)).astype(int).flatten()
    stratify_labels = data_dict["irrigation_labels"].astype(str) + "_" + y_binned.astype(str)
    stratify_encoded = LabelEncoder().fit_transform(stratify_labels)
    
    
    # -----------------------------------------------------------------
    # Train / Test Splits
    # -----------------------------------------------------------------
    indices = np.arange(len(stratify_encoded))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=args.test_size,
        stratify=stratify_encoded,
        random_state=args.seed
    )
    
    
    X = data_dict.loc[:, selected_features + ['irrigation']]
    y = data_dict['yield']
    
    
    # Standardize features based on training data
    scaler = StandardScaler()
    # Fit scaler on the training subset 
    X_train_scaled = scaler.fit_transform(X.loc[train_idx])
    # Transform the data using the training parameters
    X_scaled = scaler.transform(X)
    
    model = choose_base_model(X_train_scaled, y.loc[train_idx], args)

    # --- Make predictions ---
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_scaled[test_idx])

    # --- Calculate performance metrics ---
    train_metrics = calculate_metrics(y.loc[train_idx], y_train_pred)
    test_metrics  = calculate_metrics( y.loc[test_idx], y_test_pred)
    
    model_path = os.path.join(results_output_path, f'model_{args.seed}.pt')
    joblib.dump(model, model_path)
    
    preds_all = model.predict(X_scaled)
    save_best_model_predictions(data_dict = data_dict, 
                                train_idx = train_idx, 
                                val_idx = [], 
                                test_idx = test_idx, 
                                best_preds_all = preds_all,
                                filepath = os.path.join(results_output_path,f'predictions_{args.seed}.csv'))
    
    print("\nresults:")
    print(f"Train - MSE:{train_metrics['mse']:.4f}, MAE:{train_metrics['mae']:.4f}, R2:{train_metrics['r2']:.4f}")
    print(f"Test - MSE:{test_metrics['mse']:.4f}, MAE:{test_metrics['mae']:.4f}, R2:{test_metrics['r2']:.4f}")
    
    
def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    processor = DataProcessor(os.path.join(PROJECT_ROOT, args.data_path))
    data_dict = processor.load_and_process(args.timepoint)
    selected_features = select_features_with_high_correlation(data_dict, threshold=args.threshold)
    
    results_output_path = os.path.join(PROJECT_ROOT, args.model_type, 'results', args.method, args.timepoint)
    if not os.path.exists(results_output_path):
                os.makedirs(results_output_path)

    train_model(data_dict, selected_features, results_output_path, args)
    