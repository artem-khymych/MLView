import numpy as np
from scipy.stats import spearmanr, kendalltau
from sklearn.neighbors import NearestNeighbors
from project.logic.evaluation.metric_strategies.metric_strategy import MetricStrategy

class AnomalyDetectionMetric(MetricStrategy):
    def evaluate(self, X, anomaly_scores, y_true=None, contamination=0.1):
        """
        Evaluates anomaly detection experiments using different metrics.

        Parameters:
        -----------
        X : array-like
            Input data used for detection.
        anomaly_scores : array-like
            Anomaly scores or decision values returned by the detector.
            Higher values should indicate higher likelihood of being an anomaly.
        y_true : array-like, optional
            True labels if available (1 for anomalies, 0 for normal).
            If provided, supervised metrics will be calculated.
        contamination : float, default=0.1
            Expected proportion of anomalies in the dataset.
            Used to determine threshold when y_true is not available.

        Returns:
        -----------
        dict
            Dictionary with evaluated metrics.
        """
        metrics = {}

        # Convert scores to a consistent format (higher = more anomalous)
        # Some algorithms like LOF return negative scores for outliers
        if np.mean(anomaly_scores[anomaly_scores < 0]) < np.mean(anomaly_scores[anomaly_scores > 0]):
            anomaly_scores = -anomaly_scores

        # Threshold-based evaluation using the contamination rate if no ground truth
        if y_true is None:
            threshold = np.percentile(anomaly_scores, 100 * (1 - contamination))
            y_pred = (anomaly_scores > threshold).astype(int)

            # Unsupervised metrics
            metrics['num_detected_anomalies'] = np.sum(y_pred)
            metrics['detection_rate'] = np.sum(y_pred) / len(y_pred)

            # Add density-based metrics
            density_metrics = self._density_based_metrics(X, anomaly_scores, y_pred)
            metrics.update(density_metrics)

            # Add stability metrics
            stability_metrics = self._stability_metrics(anomaly_scores)
            metrics.update(stability_metrics)

        return metrics

    def _density_based_metrics(self, X, anomaly_scores, y_pred, k=1):
        """
        Calculate density-based metrics for evaluating anomaly detection
        without ground truth.

        Parameters:
        -----------
        X : array-like
            Input data.
        anomaly_scores : array-like
            Anomaly scores from the detector.
        y_pred : array-like
            Binary predictions based on threshold.
        k : int, default=10
            Number of neighbors for density calculation.

        Returns:
        -----------
        dict
            Dictionary with density-based metrics.
        """
        metrics = {}
        # Calculate local density using k-nearest neighbors
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(X)
        distances, _ = nn.kneighbors(X)
        local_density = 1.0 / (np.mean(distances[:, 1:], axis=1) + 1e-10)

        # Correlation between anomaly scores and local density
        # Anomalies should have lower density (higher distances to neighbors)
        corr, _ = spearmanr(-local_density, anomaly_scores)
        metrics['density_correlation'] = corr if not np.isnan(corr) else 0

        # Average relative density of detected anomalies vs normal points
        anomaly_density = local_density[y_pred == 1].mean() if np.any(y_pred == 1) else 0
        normal_density = local_density[y_pred == 0].mean() if np.any(y_pred == 0) else 0
        metrics['relative_density'] = (normal_density / (anomaly_density + 1e-10)) if anomaly_density > 0 else 1.0

        return metrics

    def _stability_metrics(self, anomaly_scores):
        """
        Calculate metrics related to the stability and distribution of scores.

        Parameters:
        -----------
        anomaly_scores : array-like
            Anomaly scores from the detector.

        Returns:
        -----------
        dict
            Dictionary with stability metrics.
        """
        metrics = {}

        # Score distribution statistics
        metrics['score_mean'] = np.mean(anomaly_scores)
        metrics['score_std'] = np.std(anomaly_scores)
        metrics['score_skewness'] = (np.mean((anomaly_scores - metrics['score_mean']) ** 3) /
                                     (metrics['score_std'] ** 3)) if metrics['score_std'] > 0 else 0

        # Score normality (higher values indicate more non-normal distribution)
        q25, q50, q75 = np.percentile(anomaly_scores, [25, 50, 75])
        iqr = q75 - q25
        metrics['score_normality'] = iqr / (2 * 0.6745 * metrics['score_std']) if metrics['score_std'] > 0 else 1.0

        # Detect bimodality in score distribution
        hist, _ = np.histogram(anomaly_scores, bins='auto')
        peaks = np.where((hist[1:-1] > hist[:-2]) & (hist[1:-1] > hist[2:]))[0] + 1
        metrics['bimodality'] = len(peaks)

        return metrics

    def compare_detectors(self, X, detector_scores_dict, y_true=None, contamination=0.1):
        """
        Compare multiple anomaly detectors.

        Parameters:
        -----------
        X : array-like
            Input data used for detection.
        detector_scores_dict : dict
            Dictionary with detector names as keys and anomaly scores as values.
        y_true : array-like, optional
            True labels if available (1 for anomalies, 0 for normal).
        contamination : float, default=0.1
            Expected proportion of anomalies in the dataset.

        Returns:
        -----------
        dict
            Dictionary with metrics for each detector.
        """
        results = {}

        for detector_name, scores in detector_scores_dict.items():
            results[detector_name] = self.evaluate(X, scores, y_true, contamination)

        # Calculate agreement between detectors if multiple detectors
        if len(detector_scores_dict) > 1:
            agreement_metrics = self._calculate_detector_agreement(detector_scores_dict)
            results['detector_agreement'] = agreement_metrics

        return results

    def _calculate_detector_agreement(self, detector_scores_dict):
        """
        Calculate agreement between different detectors.

        Parameters:
        -----------
        detector_scores_dict : dict
            Dictionary with detector names as keys and anomaly scores as values.

        Returns:
        -----------
        dict
            Dictionary with agreement metrics.
        """
        agreement = {}
        detector_names = list(detector_scores_dict.keys())

        # Calculate rank correlation between pairs of detectors
        for i in range(len(detector_names)):
            for j in range(i + 1, len(detector_names)):
                name_i = detector_names[i]
                name_j = detector_names[j]

                scores_i = detector_scores_dict[name_i]
                scores_j = detector_scores_dict[name_j]

                # Spearman rank correlation
                corr, _ = spearmanr(scores_i, scores_j)
                agreement[f'spearman_{name_i}_{name_j}'] = corr if not np.isnan(corr) else 0

                # Kendall tau rank correlation
                corr, _ = kendalltau(scores_i, scores_j)
                agreement[f'kendall_{name_i}_{name_j}'] = corr if not np.isnan(corr) else 0

                # Top-K overlap (for top 5% of anomalies)
                k = max(int(0.05 * len(scores_i)), 1)
                top_k_i = np.argsort(scores_i)[-k:]
                top_k_j = np.argsort(scores_j)[-k:]
                overlap = len(set(top_k_i).intersection(set(top_k_j)))
                agreement[f'top_k_overlap_{name_i}_{name_j}'] = overlap / k

        return agreement
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
            # Unsupervised metrics
            # For anomaly detection, num_detected_anomalies is context-dependent
            # It should match the expected contamination rate
            'num_detected_anomalies': None,  # Context-dependent
            'detection_rate': None,  # Context-dependent

            # Density-based metrics
            'density_correlation': True,  # Higher correlation between anomaly scores and low density is better
            'relative_density': True,  # Higher ratio of normal to anomaly density is better

            # Stability metrics
            'score_mean': None,  # Context-dependent
            'score_std': None,  # Context-dependent
            'score_skewness': None,  # Context-dependent
            'score_normality': True,  # Higher values indicate more non-normal distribution (better for anomaly scores)
            'bimodality': True,  # Higher number of peaks may indicate better separation

            # Agreement metrics (these will be dynamically named based on detector pairs)
            'spearman': True,  # Higher agreement between detectors is generally better
            'kendall': True,  # Higher agreement between detectors is generally better
            'top_k_overlap': True  # Higher overlap in top anomalies is better
        }

        return metainformation