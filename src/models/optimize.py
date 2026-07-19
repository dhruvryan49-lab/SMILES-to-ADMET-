import optuna
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier, XGBRegressor

class HyperparameterOptimizer:
    # 1. Added scale_pos_weight to init
    def __init__(self, task_type, random_seed=42, scale_pos_weight=1.0):
        self.task_type = task_type
        self.random_seed = random_seed
        self.scale_pos_weight = scale_pos_weight 
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def optimize(self, X, y, n_trials=15):
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 250),
                'max_depth': trial.suggest_int('max_depth', 3, 9),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 10.0, log=True), 
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 10.0, log=True),
                'random_state': self.random_seed,
                'n_jobs': -1
            }

            if self.task_type == 'classification':
                model = XGBClassifier(
                    **params, 
                    scale_pos_weight=self.scale_pos_weight, 
                    eval_metric='logloss',
                    tree_method='hist',   # Wakes up GPU optimization
                    device='cuda'         # Forces it to use the GPU
                )
                scoring = 'roc_auc'
            else:
                model = XGBRegressor(
                    **params,
                    tree_method='hist',   # Wakes up GPU optimization
                    device='cuda'         # Forces it to use the GPU
                )
                scoring = 'neg_root_mean_squared_error'

            scores = cross_val_score(model, X, y, cv=3, scoring=scoring)
            return scores.mean()

        study = optuna.create_study(direction='maximize')
        print(f"      [Optimizer] Running {n_trials} Bayesian trials...")
        study.optimize(objective, n_trials=n_trials)
        
        print(f"      [Optimizer] Best CV Score: {study.best_value:.4f}")
        return study.best_params