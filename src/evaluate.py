from sklearn.metrics import (roc_auc_score, average_precision_score, 
                             accuracy_score, mean_squared_error, r2_score)
import numpy as np

class Evaluator:
    @staticmethod
    def evaluate(y_true, y_pred, y_pred_proba, metrics_list):
        """
        Calculates only the metrics specified in the YAML config.
        """
        results = {}
        for metric in metrics_list:
            if metric == 'roc_auc':
                results['ROC-AUC'] = roc_auc_score(y_true, y_pred_proba)
            elif metric == 'auprc':
                results['AUPRC'] = average_precision_score(y_true, y_pred_proba)
            elif metric == 'accuracy':
                results['Accuracy'] = accuracy_score(y_true, y_pred)
            elif metric == 'rmse':
                results['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
            elif metric == 'r2':
                results['R2'] = r2_score(y_true, y_pred)
            else:
                print(f"Warning: Metric '{metric}' not recognized.")
        return results