# patern strategy
class MetricStrategy:
    def evaluate(self, y_true, y_pred):
        raise NotImplementedError

    def get_metainformation(self):
        raise NotImplementedError



import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)
from scipy import stats
from statsmodels.tsa.stattools import acf, pacf, adfuller, kpss
import warnings


class TimeSeriesMetric(MetricStrategy):
    def evaluate(self,y_true, y_pred, y_naive=None, residuals=None, alpha=0.05,
                 seasonality=None, freq=None, return_diagnostics=False):
        """
        Обчислює метрики для оцінки якості моделей часових рядів.

        Параметри:
        -----------
        y_true : array-like
            Справжні значення часового ряду.
        y_pred : array-like
            Прогнозовані значення часового ряду.
        y_naive : array-like, optional
            Прогноз наївної моделі (наприклад, просте зміщення на один крок).
            Якщо не вказано, використовується просте значення з попереднього кроку.
        residuals : array-like, optional
            Залишки моделі. Якщо не вказано, обчислюються як y_true - y_pred.
        alpha : float, optional (default=0.05)
            Рівень значущості для інтервалів довіри та статистичних тестів.
        seasonality : int, optional
            Період сезонності (наприклад, 12 для місячних даних з річною сезонністю).
        freq : str, optional
            Частота часового ряду для масштабування деяких метрик (наприклад, 'D' для щоденних даних).
        return_diagnostics : bool, optional (default=False)
            Якщо True, повертає додаткові діагностичні графіки та дані для залишків.

        Повертає:
        -----------
        dict
            Словник з обчисленими метриками.
        """
        metrics = {}
        y_true = [item for item in y_true]
        y_pred = [item[0] for item in y_pred]
        # Перевірка вхідних даних
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

#        y_pred = [item[0] for item in y_pred]

        if len(y_true) != len(y_pred):
            raise ValueError("y_true і y_pred повинні мати однакову довжину")

        # 1. Базові метрики точності прогнозування

        # Середня абсолютна похибка
        metrics['mae'] = mean_absolute_error(y_true, y_pred)

        # Середньоквадратична похибка
        metrics['mse'] = mean_squared_error(y_true, y_pred)

        # Корінь із середньоквадратичної похибки
        metrics['rmse'] = np.sqrt(metrics['mse'])

        # Середня абсолютна процентна похибка
        # Уникаємо ділення на нуль
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred) * 100
            except:
                nonzero_idx = y_true != 0
                if np.any(nonzero_idx):
                    metrics['mape'] = np.mean(
                        np.abs((y_true[nonzero_idx] - y_pred[nonzero_idx]) / y_true[nonzero_idx])) * 100
                else:
                    metrics['mape'] = np.nan

        # Симетрична MAPE (sMAPE)
        denominator = np.abs(y_true) + np.abs(y_pred)
        nonzero_idx = denominator != 0
        if np.any(nonzero_idx):
            metrics['smape'] = np.mean(
                2 * np.abs(y_true[nonzero_idx] - y_pred[nonzero_idx]) / denominator[nonzero_idx]) * 100
        else:
            metrics['smape'] = np.nan

        # Середня абсолютна масштабована похибка (MASE)
        # Потрібен базовий наївний прогноз для масштабування
        if y_naive is None and len(y_true) > 1:
            # Використовуємо просту модель: значення з попереднього кроку
            y_naive = np.concatenate([[y_true[0]], y_true[:-1]])

        if y_naive is not None:
            naive_errors = np.abs(y_true - y_naive)
            if np.sum(naive_errors) > 0:
                metrics['mase'] = np.mean(np.abs(y_true - y_pred)) / np.mean(naive_errors)
            else:
                metrics['mase'] = np.nan

        # Коефіцієнт детермінації R²
        metrics['r2'] = r2_score(y_true, y_pred)

        # 2. Аналіз залишків

        if residuals is None:
            residuals = y_true - y_pred

        # Середнє значення залишків (має бути близьким до нуля)
        metrics['residual_mean'] = np.mean(residuals)

        # Стандартне відхилення залишків
        metrics['residual_std'] = np.std(residuals)

        # Тест Дікі-Фуллера для стаціонарності залишків
        try:
            adf_result = adfuller(residuals)
            metrics['adf_statistic'] = adf_result[0]
            metrics['adf_pvalue'] = adf_result[1]
            metrics['residuals_stationary'] = adf_result[1] < alpha
        except:
            metrics['adf_statistic'] = np.nan
            metrics['adf_pvalue'] = np.nan
            metrics['residuals_stationary'] = None

        # Тест Квятковського-Філліпса-Шмідта-Шина (KPSS) для стаціонарності
        try:
            kpss_result = kpss(residuals)
            metrics['kpss_statistic'] = kpss_result[0]
            metrics['kpss_pvalue'] = kpss_result[1]
            metrics['residuals_trend_stationary'] = kpss_result[1] > alpha
        except:
            metrics['kpss_statistic'] = np.nan
            metrics['kpss_pvalue'] = np.nan
            metrics['residuals_trend_stationary'] = None

        # Тест Бокса-Пірса для автокореляції
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb_result = acorr_ljungbox(residuals, lags=min(10, len(residuals) // 5))
            metrics['ljung_box_statistic'] = lb_result.iloc[-1, 0]
            metrics['ljung_box_pvalue'] = lb_result.iloc[-1, 1]
            metrics['residuals_independent'] = lb_result.iloc[-1, 1] > alpha
        except:
            metrics['ljung_box_statistic'] = np.nan
            metrics['ljung_box_pvalue'] = np.nan
            metrics['residuals_independent'] = None

        # Тест Харке-Бера для нормальності
        try:
            jb_stat, jb_pvalue = stats.jarque_bera(residuals)
            metrics['jarque_bera_statistic'] = jb_stat
            metrics['jarque_bera_pvalue'] = jb_pvalue
            metrics['residuals_normal'] = jb_pvalue > alpha
        except:
            metrics['jarque_bera_statistic'] = np.nan
            metrics['jarque_bera_pvalue'] = np.nan
            metrics['residuals_normal'] = None

        # Автокореляція залишків
        try:
            residual_acf = acf(residuals, nlags=min(20, len(residuals) // 4), fft=True)
            # Кількість значущих лагів автокореляції
            confidence_interval = stats.norm.ppf(1 - alpha / 2) / np.sqrt(len(residuals))
            significant_lags = np.sum(np.abs(residual_acf[1:]) > confidence_interval)
            metrics['significant_acf_lags'] = significant_lags

            # Часткова автокореляція залишків
            residual_pacf = pacf(residuals, nlags=min(20, len(residuals) // 4))
            # Кількість значущих лагів часткової автокореляції
            significant_pacf_lags = np.sum(np.abs(residual_pacf[1:]) > confidence_interval)
            metrics['significant_pacf_lags'] = significant_pacf_lags
        except:
            metrics['significant_acf_lags'] = np.nan
            metrics['significant_pacf_lags'] = np.nan

        # 3. Спеціалізовані метрики для часових рядів

        # Тіл U статистика (Theil's U) - порівнює з наївним прогнозом
        if y_naive is not None:
            naive_mse = mean_squared_error(y_true, y_naive)
            if naive_mse > 0:
                metrics['theils_u'] = np.sqrt(metrics['mse'] / naive_mse)
            else:
                metrics['theils_u'] = np.nan

        # Оцінка точності напрямку (Direction Accuracy)
        if len(y_true) > 1:
            actual_direction = np.sign(y_true[1:] - y_true[:-1])
            pred_direction = np.sign(y_pred[1:] - y_pred[:-1])
            metrics['direction_accuracy'] = np.mean(actual_direction == pred_direction) * 100

        # Пікова відносна помилка (PRE)
        peak_idx = np.argmax(np.abs(y_true))
        metrics['peak_error'] = np.abs(y_true[peak_idx] - y_pred[peak_idx])
        if y_true[peak_idx] != 0:
            metrics['peak_relative_error'] = metrics['peak_error'] / np.abs(y_true[peak_idx]) * 100
        else:
            metrics['peak_relative_error'] = np.nan

        # 4. Сезонні метрики (якщо вказано сезонність)
        if seasonality is not None and len(y_true) >= 2 * seasonality:
            # Розділити дані на сезонні компоненти
            n_seasons = len(y_true) // seasonality
            seasonal_errors = np.zeros(seasonality)

            for i in range(seasonality):
                season_indices = [i + s * seasonality for s in range(n_seasons) if i + s * seasonality < len(y_true)]
                if season_indices:
                    seasonal_errors[i] = np.mean(np.abs(y_true[season_indices] - y_pred[season_indices]))

            metrics['max_seasonal_error'] = np.max(seasonal_errors)
            metrics['min_seasonal_error'] = np.min(seasonal_errors)
            metrics['seasonal_error_std'] = np.std(seasonal_errors)
            metrics['seasonal_error_ratio'] = metrics['max_seasonal_error'] / metrics['min_seasonal_error'] if metrics[
                                                                                                                   'min_seasonal_error'] > 0 else np.nan

        # 5. Додаткові діагностичні дані
        if return_diagnostics:
            diagnostics = {}

            # ACF і PACF залишків
            try:
                diagnostics['residual_acf'] = acf(residuals, nlags=min(40, len(residuals) // 2), fft=True)
                diagnostics['residual_pacf'] = pacf(residuals, nlags=min(40, len(residuals) // 2))
            except:
                diagnostics['residual_acf'] = None
                diagnostics['residual_pacf'] = None

            # QQ-plot дані
            try:
                from scipy.stats import probplot
                qq_data = probplot(residuals, dist='norm')
                diagnostics['qq_plot_data'] = qq_data
            except:
                diagnostics['qq_plot_data'] = None

            # Розподіл помилок за величиною
            error_bins = np.histogram(np.abs(residuals), bins=10)
            diagnostics['error_distribution'] = error_bins

            # Повертаємо обидва результати - метрики та діагностику
            return metrics, diagnostics

        return metrics
