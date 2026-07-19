import yaml
import pandas as pd
import numpy as np
import warnings
import os
import joblib
from rdkit import RDLogger

# Import our custom modules
from src.data_loader import ADMETDataLoader
from src.features import FeatureExtractor
from src.preprocessing import Preprocessor
from src.models import ModelFactory, HyperparameterOptimizer
from src.evaluate import Evaluator

# Suppress warnings for clean terminal output
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')

def main():
    print("=== STARTING MODULAR ADMET PIPELINE (WITH OPTUNA) ===")
    
    os.makedirs('outputs/models', exist_ok=True)
    os.makedirs('outputs/results', exist_ok=True)
    
    with open('configs/endpoints.yaml', 'r') as file:
        config = yaml.safe_load(file)
        
    feature_extractor = FeatureExtractor(config)
    smiles_col = config['features']['smiles_column']
    
    final_results = []

    for endpoint_name, endpoint_config in config['endpoints'].items():
        print(f"\n-> Processing: {endpoint_name}")
        task_type = endpoint_config['task_type']
        
        # A. Data Loading & Splitting
        loader = ADMETDataLoader(config, endpoint_name)
        train_df, valid_df, test_df = loader.get_data_splits()
        
        # B. Feature Extraction
        print("   Extracting high-resolution molecular features...")
        for df in [train_df, valid_df, test_df]:
            df['Features'] = df[smiles_col].apply(feature_extractor.extract)
            
        # C. Preprocessing
        train_df = Preprocessor.clean_data(train_df, feature_column='Features')
        valid_df = Preprocessor.clean_data(valid_df, feature_column='Features')
        test_df = Preprocessor.clean_data(test_df, feature_column='Features')
        
        # D. Matrix Stacking
        X_train = np.vstack(train_df['Features'].values)
        y_train = train_df[endpoint_config['data']['target_column']].values
        
        X_test = np.vstack(test_df['Features'].values)
        y_test = test_df[endpoint_config['data']['target_column']].values
        
        active_model_name = endpoint_config['active_model']
        
        # --- NEW: Dynamic Class Weighting ---
        if task_type == 'classification':
            num_negative = np.sum(y_train == 0)
            num_positive = np.sum(y_train == 1)
            
            imbalance_ratio = num_negative / num_positive if num_positive > 0 else 1.0
            print(f"   [Class Weights] Neg: {num_negative} | Pos: {num_positive} -> scale_pos_weight: {imbalance_ratio:.2f}")
            
            # Inject directly into the model's configuration
            config['models'][active_model_name]['hyperparameters']['scale_pos_weight'] = float(imbalance_ratio)
            
        # E. OPTUNA BAYESIAN OPTIMIZATION
        if config['models'][active_model_name]['algorithm'] == 'xgboost':
            print(f"   Optimizing hyperparameters for {task_type}...")
            
            # Fetch the calculated weight (defaults to 1.0 for regression)
            spw = config['models'][active_model_name]['hyperparameters'].get('scale_pos_weight', 1.0)
            
            optimizer = HyperparameterOptimizer(
                task_type=task_type, 
                random_seed=config['global']['random_seed'],
                scale_pos_weight=spw  # Pass it to Optuna
            )
            
            best_params = optimizer.optimize(X_train, y_train, n_trials=15)
            config['models'][active_model_name]['hyperparameters'].update(best_params)

        # F. Model Initialization & Training
        model = ModelFactory.build_model(config, active_model_name, task_type)
        
        print(f"   Training final optimized {active_model_name}...")
        model.fit(X_train, y_train)
        
        # G. Persistence
        model_path = f"outputs/models/{endpoint_name}_{active_model_name}.pkl"
        joblib.dump(model, model_path)
        
        # H. Inference & Metrics Calculation
        print("   Evaluating on test set...")
        if task_type == 'classification':
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)
        else:
            y_pred = model.predict(X_test)
            y_pred_proba = None 
            
        metrics_to_calculate = endpoint_config['evaluation']['metrics']
        scores = Evaluator.evaluate(y_test, y_pred, y_pred_proba, metrics_to_calculate)
        
        result_row = {'Endpoint': endpoint_name, 'Task': task_type.capitalize()}
        for metric_name, score in scores.items():
            result_row[metric_name] = round(score, 3)
        final_results.append(result_row)
        print(f"   Scores: {scores}")

    # Final Leaderboard Generation
    print("\n" + "="*70)
    print(" FINAL MODULAR EVALUATION REPORT")
    print("="*70)
    results_df = pd.DataFrame(final_results)
    
    results_path = 'outputs/results/leaderboard.csv'
    results_df.to_csv(results_path, index=False)
    
    results_df = results_df.fillna("") 
    print(results_df.to_string(index=False))
    print("="*70)

if __name__ == "__main__":
    main()