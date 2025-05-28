import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    r2_score, median_absolute_error,
    explained_variance_score, max_error
)

from project.logic.evaluation.metric_strategies.metric_strategy import MetricStrategy


class RegressionMetric(MetricStrategy):
    def evaluate(self, y_true, y_pred):
        """
        Ecaluates the regression experiment using different metrics.

        :param:
        -----------
        y_true : array-like
            true values of target variable
        y_pred : array-like
            predicted values of target variable.

        :returns:
        -----------
        dict
            Dictionary with evaluated metrics.
        """
        metrics = {}

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Basic metrics
        metrics['mse'] = mean_squared_error(y_true, y_pred)  # Mean Squared Error
        metrics['rmse'] = np.sqrt(metrics['mse'])  # Root Mean Squared Error
        metrics['mae'] = mean_absolute_error(y_true, y_pred)  # Mean Absolute Error
        metrics['medae'] = median_absolute_error(y_true, y_pred)  # Median Absolute Error
        metrics['max_error'] = max_error(y_true, y_pred)  # Maximum Error

        # Relative error metrics
        metrics['mape'] = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100  # Mean Absolute Percentage Error
        metrics['smape'] = np.mean(
            2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-10)) * 100  # Symmetric MAPE

        # Quality metrics
        metrics['r2'] = r2_score(y_true, y_pred)  # R-squared (Coefficient of Determination)
        metrics['adjusted_r2'] = 1 - (1 - metrics['r2']) * (len(y_true) - 1) / (
            len(y_true) - y_pred.shape[1] if y_pred.ndim > 1 else len(y_true) - 1)  # Adjusted R-squared
        metrics['explained_variance'] = explained_variance_score(y_true, y_pred)  # Explained Variance

        errors = y_true - y_pred
        metrics['mean_error'] = np.mean(errors)  # Mean Error (ME)
        metrics['std_error'] = np.std(errors)  # Standard Deviation of Errors

        # Normalized error metrics
        y_true_var = np.var(y_true)
        if y_true_var > 0:
            metrics['nmse'] = metrics['mse'] / y_true_var  # Normalized Mean Squared Error
            metrics['nrmse'] = metrics['rmse'] / np.mean(y_true)  # Normalized Root Mean Squared Error
            metrics['rrse'] = np.sqrt(
                np.sum(np.square(errors)) / np.sum(np.square(y_true - np.mean(y_true))))  # Root Relative Squared Error
        else:
            metrics['nmse'] = float('inf')
            metrics['nrmse'] = float('inf')
            metrics['rrse'] = float('inf')

        # Theil's U statistic (version 2)
        sum_squared_pred = np.sum(np.square(y_pred))
        sum_squared_true = np.sum(np.square(y_true))
        if sum_squared_pred > 0 and sum_squared_true > 0:
            metrics['theil_u2'] = np.sqrt(np.sum(np.square(y_true - y_pred)) / sum_squared_true)
        else:
            metrics['theil_u2'] = float('inf')

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
            # Error metrics (lower is better)
            'mse': False,
            'rmse': False,
            'mae': False,
            'medae': False,
            'max_error': False,
            'mape': False,
            'smape': False,
            'mean_error': None,  # Ideally should be close to zero, not strictly minimized or maximized
            'std_error': False,
            'nmse': False,
            'nrmse': False,
            'rrse': False,
            'theil_u2': False,

            # Quality metrics (higher is better)
            'r2': True,
            'adjusted_r2': True,
            'explained_variance': True
        }

        return metainformation

