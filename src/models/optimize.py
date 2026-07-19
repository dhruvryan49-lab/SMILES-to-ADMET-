import optuna
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier, XGBRegressor

class HyperparameterOptimizer:
    def __init__(self, task_type, random_seed=42):
        self.task_type = task_type
        self.random_seed = random_seed
        # Keep the terminal clean from hundreds of Optuna logs
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def optimize(self, X, y, n_trials=15):
        """
        Runs Bayesian optimization to find the best hyperparameters,
        specifically targeting regularization to prevent overfitting.
        """
        def objective(trial):
            # The search space for our XGBoost model
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 250),
                'max_depth': trial.suggest_int('max_depth', 3, 9),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                # Crucial for 2048 bits: L1 (Lasso) and L2 (Ridge) Regularization
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 10.0, log=True), 
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 10.0, log=True),
                'random_state': self.random_seed,
                'n_jobs': -1
            }

            if self.task_type == 'classification':
                model = XGBClassifier(**params, eval_metric='logloss')
                scoring = 'roc_auc'
            else:
                model = XGBRegressor(**params)
                scoring = 'neg_root_mean_squared_error'

            # 3-fold cross-validation to ensure the parameters actually generalize
            scores = cross_val_score(model, X, y, cv=3, scoring=scoring)
            return scores.mean()

        study = optuna.create_study(direction='maximize')
        print(f"      [Optimizer] Running {n_trials} Bayesian trials...")
        study.optimize(objective, n_trials=n_trials)
        
        print(f"      [Optimizer] Best CV Score: {study.best_value:.4f}")
        return study.best_params