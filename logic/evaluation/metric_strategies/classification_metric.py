import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score, log_loss, cohen_kappa_score,
    matthews_corrcoef, balanced_accuracy_score
)

from project.logic.evaluation.metric_strategies.metric_strategy import MetricStrategy


class ClassificationMetric(MetricStrategy):
    def evaluate(self, y_true, y_pred, y_prob=None):
        """
        Ecaluates the classification experiment using different metrics.

        :param:
        -----------
        y_true : array-like
            True labels for classes.
        y_pred : array-like
            Predicted labels for classes.
        y_prob : array-like, optional
            Probability of predicted true clss, needed for
            ROC AUC, PR AUC and log loss. Optional

        :returns:
        -----------
        dict
            Dictionary with evaluated metrics.
        """
        metrics = {}

        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)

        cm = confusion_matrix(y_true, y_pred)
        # metrics['confusion_matrix'] = cm

        # for binary classification
        if len(np.unique(y_true)) == 2:
            metrics['precision'] = precision_score(y_true, y_pred, average='binary')
            metrics['recall'] = recall_score(y_true, y_pred, average='binary')
            metrics['f1'] = f1_score(y_true, y_pred, average='binary')
            metrics['specificity'] = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0

            #
            if y_prob is not None:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
                metrics['pr_auc'] = average_precision_score(y_true, y_prob)
                metrics['log_loss'] = log_loss(y_true, y_prob)

        # for multi class classification
        else:
            metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro')
            metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro')
            metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted')

            metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro')
            metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro')
            metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted')

            metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
            metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro')
            metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')

            # metrics['classification_report'] = classification_report(y_true, y_pred)
            if y_prob is not None and hasattr(y_prob, 'shape') and len(y_prob.shape) > 1:
                try:
                    metrics['log_loss'] = log_loss(y_true, y_prob)
                except:
                    pass

        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)

        return metrics

    def get_metainformation(self):
        """
        Returns a dictionary with information about metrics optimization direction.
        For each metric, indicates whether higher (True) or lower (False) values
        are better.

        :returns:
        -----------
        dict
            Dictionary with metric names as keys and boolean values indicating
            if higher values are better (True) or lower values are better (False).
        """
        metainformation = {
            # Higher is better metrics (True)
            'accuracy': True,
            'balanced_accuracy': True,
            'precision': True,
            'recall': True,
            'f1': True,
            'specificity': True,
            'roc_auc': True,
            'pr_auc': True,
            'precision_macro': True,
            'precision_micro': True,
            'precision_weighted': True,
            'recall_macro': True,
            'recall_micro': True,
            'recall_weighted': True,
            'f1_macro': True,
            'f1_micro': True,
            'f1_weighted': True,
            'cohen_kappa': True,
            'matthews_corrcoef': True,

            # Lower is better metrics (False)
            'log_loss': False
        }

        return metainformation