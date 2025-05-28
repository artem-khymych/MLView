import numpy as np
from sklearn.metrics import (
    mean_squared_error, explained_variance_score
)
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from sklearn.manifold import trustworthiness

from project.logic.evaluation.metric_strategies.metric_strategy import MetricStrategy


class DimReduction(MetricStrategy):
    def evaluate(self, X_original, X_reduced, X_reconstructed=None, n_neighbors=5,
                 precomputed_distances=None, precomputed_reduced_distances=None):
        """
        Обчислює різні метрики для оцінки якості зменшення розмірності.

        Параметри:
        -----------
        X_original : array-like
            Оригінальні дані високої розмірності.
        X_reduced : array-like
            Дані зниженої розмірності.
        X_reconstructed : array-like, optional
            Реконструйовані дані (з зниженої розмірності назад у вихідну).
            Потрібно для обчислення помилки реконструкції.
        n_neighbors : int, optional (default=5)
            Кількість сусідів для обчислення метрик збереження сусідства.
        precomputed_distances : array-like, optional
            Попередньо обчислена матриця відстаней для оригінальних даних.
        precomputed_reduced_distances : array-like, optional
            Попередньо обчислена матриця відстаней для даних зниженої розмірності.

        Повертає:
        -----------
        dict
            Словник з обчисленими метриками.
        """
        metrics = {}

        # Перевірка вхідних даних
        X_original = np.array(X_original)
        X_reduced = np.array(X_reduced)

        # Обчислення матриць відстаней, якщо вони не передані
        if precomputed_distances is None:
            dist_original = squareform(pdist(X_original))
        else:
            dist_original = precomputed_distances

        if precomputed_reduced_distances is None:
            dist_reduced = squareform(pdist(X_reduced))
        else:
            dist_reduced = precomputed_reduced_distances

        # 1. Кореляція Пірсона між відстанями
        # Перетворимо матриці відстаней у вектори (беремо верхні трикутні частини)
        n = dist_original.shape[0]
        triu_indices = np.triu_indices(n, k=1)
        dist_original_vec = dist_original[triu_indices]
        dist_reduced_vec = dist_reduced[triu_indices]

        # Обчислення кореляції
        pearson_corr, _ = pearsonr(dist_original_vec, dist_reduced_vec)
        metrics['pearson_correlation'] = pearson_corr

        # 2. Кореляція Спірмена між відстанями (ранговий коефіцієнт)
        spearman_corr, _ = spearmanr(dist_original_vec, dist_reduced_vec)
        metrics['spearman_correlation'] = spearman_corr

        # 3. Коефіцієнт надійності (Trustworthiness)
        # Вимірює, наскільки зберігаються локальні сусідства
        try:
            trust = trustworthiness(X_original, X_reduced, n_neighbors=n_neighbors)
            metrics['trustworthiness'] = trust
        except:
            metrics['trustworthiness'] = float('nan')

        # 4. Continuity (доповнення до Trustworthiness)
        # Continuity вимірює, чи нові сусіди в скороченому просторі
        # є справжніми сусідами в оригінальному просторі
        try:
            continuity = trustworthiness(X_reduced, X_original, n_neighbors=n_neighbors)
            metrics['continuity'] = continuity
        except:
            metrics['continuity'] = float('nan')

        # 5. Відсоток збереження дисперсії (для лінійних методів)
        # Якщо X_reconstructed не передано, можемо обчислити тільки
        # якщо розмірність X_reduced < X_original
        if X_original.shape[1] > X_reduced.shape[1]:
            # Дисперсія в зниженому просторі / дисперсія в оригінальному просторі
            var_original = np.sum(np.var(X_original, axis=0))
            var_reduced = np.sum(np.var(X_reduced, axis=0))
            metrics['variance_retention_ratio'] = var_reduced / var_original

        # 6. Метрики реконструкції (якщо надано реконструйовані дані)
        if X_reconstructed is not None:
            X_reconstructed = np.array(X_reconstructed)

            # Середньоквадратична помилка реконструкції
            mse = mean_squared_error(X_original, X_reconstructed)
            metrics['reconstruction_mse'] = mse
            metrics['reconstruction_rmse'] = np.sqrt(mse)

            # Відносна помилка реконструкції
            metrics['relative_reconstruction_error'] = np.sum((X_original - X_reconstructed) ** 2) / np.sum(
                X_original ** 2)

            # Коефіцієнт поясненої дисперсії
            metrics['explained_variance'] = explained_variance_score(X_original, X_reconstructed)

        # 7. K-найближчий сусід збереження (KNN Preservation)
        def knn_preservation(dist_orig, dist_red, k):
            """Обчислює відсоток k-найближчих сусідів, які зберігаються."""
            n = dist_orig.shape[0]
            knn_orig = np.argsort(dist_orig, axis=1)[:, 1:k + 1]  # Виключаємо саму точку
            knn_red = np.argsort(dist_red, axis=1)[:, 1:k + 1]

            preservation = 0
            for i in range(n):
                intersection = np.intersect1d(knn_orig[i], knn_red[i])
                preservation += len(intersection) / k

            return preservation / n

        # Обчислюємо збереження KNN для різних значень k
        for k in [5, 10, 20]:
            if n > k:  # Перевіряємо, що у нас достатньо даних
                knn_pres = knn_preservation(dist_original, dist_reduced, k)
                metrics[f'knn_preservation_{k}'] = knn_pres

        # 8. Відношення стресу (Stress Ratio)
        # Нормалізована сума квадратів різниць між відстанями
        sum_sq_dist_original = np.sum(dist_original ** 2)
        if sum_sq_dist_original > 0:
            stress = np.sqrt(np.sum((dist_original - dist_reduced) ** 2) / sum_sq_dist_original)
            metrics['stress_ratio'] = stress

        # 9. Локальна структура (обчислення для невеликих даних, оскільки це обчислювально інтенсивно)
        if n <= 1000:
            # Локальне масштабування відстаней Самона (LSDS - Local Scaling of Distances Score)
            # Високі значення означають краще збереження локальної структури
            def local_scaling(dist_mat, k=5):
                sigma = np.zeros(n)
                for i in range(n):
                    # k+1-й найближчий сусід (включаючи саму точку)
                    kth_distance = np.sort(dist_mat[i])[min(k + 1, n - 1)]
                    sigma[i] = kth_distance

                scaled_dist = np.zeros_like(dist_mat)
                for i in range(n):
                    for j in range(n):
                        scaled_dist[i, j] = dist_mat[i, j] / (sigma[i] * sigma[j])

                return scaled_dist

            try:
                scaled_original = local_scaling(dist_original, k=n_neighbors)
                scaled_reduced = local_scaling(dist_reduced, k=n_neighbors)

                # Кореляція між масштабованими відстанями
                scaled_original_vec = scaled_original[triu_indices]
                scaled_reduced_vec = scaled_reduced[triu_indices]

                lsds_corr, _ = pearsonr(scaled_original_vec, scaled_reduced_vec)
                metrics['local_scaling_correlation'] = lsds_corr
            except:
                metrics['local_scaling_correlation'] = float('nan')

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
            # Correlation metrics - higher is better
            'pearson_correlation': True,
            'spearman_correlation': True,

            # Neighborhood preservation metrics - higher is better
            'trustworthiness': True,
            'continuity': True,

            # Variance retention - higher is better
            'variance_retention_ratio': True,

            # Reconstruction error metrics - lower is better
            'reconstruction_mse': False,
            'reconstruction_rmse': False,
            'relative_reconstruction_error': False,

            # Explained variance - higher is better
            'explained_variance': True,

            # KNN preservation metrics - higher is better
            'knn_preservation_5': True,
            'knn_preservation_10': True,
            'knn_preservation_20': True,

            # Stress ratio - lower is better
            'stress_ratio': False,

            # Local structure preservation - higher is better
            'local_scaling_correlation': True
        }

        return metainformation
