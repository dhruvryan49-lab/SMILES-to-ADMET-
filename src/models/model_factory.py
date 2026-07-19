from xgboost import XGBClassifier, XGBRegressor

class ModelFactory:
    @staticmethod
    def build_model(config, model_name, task_type):
        """
        Dynamically instantiates a model based on the YAML configuration.
        """
        model_config = config['models'][model_name]
        algo = model_config['algorithm']
        params = model_config['hyperparameters'].copy()
        
        # Inject global settings
        params['random_state'] = config['global']['random_seed']
        params['n_jobs'] = config['global']['n_jobs']
        
        if algo == 'xgboost':
            if task_type == 'classification':
                # eval_metric='logloss' suppresses an XGBoost warning
                return XGBClassifier(**params, eval_metric='logloss')
            elif task_type == 'regression':
                return XGBRegressor(**params)
        elif algo == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            if task_type == 'classification':
                return RandomForestClassifier(**params)
            return RandomForestRegressor(**params)
            
        raise ValueError(f"Algorithm {algo} not supported by ModelFactory.")