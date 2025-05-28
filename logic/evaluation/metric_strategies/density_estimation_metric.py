import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from scipy.stats import entropy
from sklearn.decomposition import PCA
from project.logic.evaluation.metric_strategies.metric_strategy import MetricStrategy


class DensityEstimationMetric(MetricStrategy):
    def evaluate(self, X, model, y_true=None, intrinsic_dim=None):
        """
        Evaluates dimensionality estimation models using different metrics.

        Parameters:
        -----------
        X : array-like
            Input data used for dimensionality estimation.
        model : object
            Fitted model (GaussianMixture or BayesianGaussianMixture).
        y_true : array-like, optional
            True cluster labels if available.
        intrinsic_dim : int, optional
            True intrinsic dimensionality if known.

        Returns:
        -----------
        dict
            Dictionary with evaluated metrics.
        """
        metrics = {}

        # Get basic model information
        n_components = model.n_components
        metrics['n_components'] = n_components

        # Get predicted labels
        y_pred = model.predict(X)

        # Get probabilities and responsibilities
        probs = model.predict_proba(X)

        # Get BIC and AIC scores which are model selection criteria
        metrics['bic'] = model.bic(X)
        metrics['aic'] = model.aic(X)

        # Calculate cluster quality metrics
        try:
            metrics['silhouette_score'] = silhouette_score(X, y_pred)
        except:
            metrics['silhouette_score'] = 0

        try:
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(X, y_pred)
        except:
            metrics['calinski_harabasz_score'] = 0

        try:
            metrics['davies_bouldin_score'] = davies_bouldin_score(X, y_pred)
        except:
            metrics['davies_bouldin_score'] = 0

        # Calculate log-likelihood
        metrics['log_likelihood'] = model.score(X) * len(X)

        # Calculate entropy of cluster assignments
        cluster_probs = np.bincount(y_pred) / len(y_pred)
        metrics['cluster_entropy'] = entropy(cluster_probs)

        # Average uncertainty in assignments
        assignment_entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
        metrics['avg_assignment_entropy'] = np.mean(assignment_entropy)

        # For BayesianGaussianMixture, check effective number of components
        if hasattr(model, 'weights_'):
            # Number of effective components (those with non-negligible weight)
            effective_n_components = np.sum(model.weights_ > 0.01)
            metrics['effective_n_components'] = effective_n_components

            # Calculate Dirichlet concentration parameter if available
            if hasattr(model, 'concentrations_'):
                metrics['dirichlet_concentration'] = np.sum(model.concentrations_)

        # Dimensionality-related metrics
        # 1. Average covariance dimensionality
        dim_metrics = self._calculate_dimensionality_metrics(X, model)
        metrics.update(dim_metrics)

        # If true intrinsic dimensionality is known, calculate error
        if intrinsic_dim is not None:
            metrics['dim_error'] = abs(metrics['estimated_intrinsic_dim'] - intrinsic_dim)
            metrics['dim_error_ratio'] = metrics['dim_error'] / intrinsic_dim if intrinsic_dim > 0 else 0

        return metrics

    def _calculate_dimensionality_metrics(self, X, model, variance_threshold=0.95):
        """
        Calculate dimensionality-related metrics based on the covariance structure.

        Parameters:
        -----------
        X : array-like
            Input data.
        model : object
            Fitted model (GaussianMixture or BayesianGaussianMixture).
        variance_threshold : float, default=0.95
            Threshold for explained variance to determine dimensionality.

        Returns:
        -----------
        dict
            Dictionary with dimensionality metrics.
        """
        metrics = {}

        # Determine data dimensionality
        original_dim = X.shape[1]
        metrics['original_dim'] = original_dim

        # Get covariances from model
        if hasattr(model, 'covariances_'):
            covariances = model.covariances_
            weights = model.weights_

            # Calculate average eigenvalues across components
            avg_eigenvalues = np.zeros(original_dim)
            for i, (cov, weight) in enumerate(zip(covariances, weights)):
                if weight > 0.01:  # Only consider components with significant weight
                    # Handle different covariance types
                    if model.covariance_type == 'full':
                        eigenvalues = np.linalg.eigvalsh(cov)
                        avg_eigenvalues += weight * eigenvalues
                    elif model.covariance_type == 'tied':
                        eigenvalues = np.linalg.eigvalsh(cov)
                        avg_eigenvalues += eigenvalues
                    elif model.covariance_type == 'diag':
                        avg_eigenvalues += weight * cov
                    elif model.covariance_type == 'spherical':
                        avg_eigenvalues += weight * np.ones(original_dim) * cov

            # Sort eigenvalues in descending order
            avg_eigenvalues = np.sort(avg_eigenvalues)[::-1]

            # Normalize eigenvalues
            total_variance = np.sum(avg_eigenvalues)
            if total_variance > 0:
                normalized_eigenvalues = avg_eigenvalues / total_variance

                # Calculate explained variance ratio
                explained_variance_ratio = np.cumsum(normalized_eigenvalues)

                # Find number of dimensions needed to explain variance_threshold of variance
                dims_needed = np.argmax(explained_variance_ratio >= variance_threshold) + 1
                metrics['estimated_intrinsic_dim'] = dims_needed

                # Calculate eigenvalue decay rate (higher means faster decay, more concentrated info)
                if len(avg_eigenvalues) > 1:
                    decay_rate = avg_eigenvalues[0] / avg_eigenvalues[-1]
                    metrics['eigenvalue_decay_rate'] = decay_rate

                # Effective rank (a continuous measure of dimensionality)
                effective_rank = np.exp(entropy(normalized_eigenvalues))
                metrics['effective_rank'] = effective_rank

                # Store eigenvalue distribution statistics
                metrics['max_eigenvalue'] = avg_eigenvalues[0]
                metrics['min_eigenvalue'] = avg_eigenvalues[-1]
                metrics['eigenvalue_ratio'] = metrics['max_eigenvalue'] / (metrics['min_eigenvalue'] + 1e-10)
        else:
            # If covariance is not available, use PCA as backup
            pca = PCA().fit(X)
            explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
            dims_needed = np.argmax(explained_variance_ratio >= variance_threshold) + 1
            metrics['estimated_intrinsic_dim'] = dims_needed
            metrics['effective_rank'] = np.exp(entropy(pca.explained_variance_ratio_))

        # Compare with MLE estimation (maximum likelihood estimator)
        try:
            mle_dim = self._mle_intrinsic_dimension(X)
            metrics['mle_intrinsic_dim'] = mle_dim
        except:
            metrics['mle_intrinsic_dim'] = original_dim

        return metrics

    def _mle_intrinsic_dimension(self, X, k=10):
        """
        Estimate intrinsic dimensionality using maximum likelihood estimation.
        Based on Levina and Bickel (2005).

        Parameters:
        -----------
        X : array-like
            Input data.
        k : int, default=10
            Number of nearest neighbors to use.

        Returns:
        -----------
        float
            Estimated intrinsic dimensionality.
        """
        # Compute k-nearest neighbors
        nn = NearestNeighbors(n_neighbors=k + 1)
        nn.fit(X)
        distances, _ = nn.kneighbors(X)

        # Remove self-distance (first column)
        distances = distances[:, 1:]

        # Calculate MLE estimate
        inv_mle = 0
        for i in range(len(X)):
            r_ik = distances[i, -1]  # Distance to k-th neighbor
            inv_mle += np.mean(np.log(r_ik / distances[i, :-1]))

        # MLE estimate for intrinsic dimensionality
        if inv_mle > 0:
            mle_dim = (len(X) * (k - 1)) / inv_mle
        else:
            mle_dim = X.shape[1]  # Default to original dimensionality

        return mle_dim

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
            # Model information metrics (context dependent)
            'n_components': None,  # Context dependent
            'effective_n_components': None,  # Context dependent

            # Information criteria (lower is better)
            'bic': False,
            'aic': False,

            # Cluster quality metrics
            'silhouette_score': True,  # Higher is better
            'calinski_harabasz_score': True,  # Higher is better
            'davies_bouldin_score': False,  # Lower is better

            # Log-likelihood (higher is better)
            'log_likelihood': True,

            # Entropy metrics (context dependent)
            'cluster_entropy': None,  # Context dependent - depends on expected number of clusters
            'avg_assignment_entropy': False,  # Lower is better - indicates more confident assignments

            # Concentration parameter (context dependent)
            'dirichlet_concentration': None,  # Context dependent

            # Dimensionality metrics
            'original_dim': None,  # Context dependent - just informational
            'estimated_intrinsic_dim': None,  # Context dependent - should match true intrinsic dim
            'mle_intrinsic_dim': None,  # Context dependent - should match true intrinsic dim
            'effective_rank': None,  # Context dependent - should match true intrinsic dim

            # Dimensionality errors (lower is better)
            'dim_error': False,
            'dim_error_ratio': False,

            # Eigenvalue metrics
            'eigenvalue_decay_rate': True,  # Higher is better - indicates more concentrated information
            'max_eigenvalue': True,  # Higher is better - indicates stronger principal direction
            'min_eigenvalue': None,  # Context dependent
            'eigenvalue_ratio': True  # Higher is better - indicates better separation of signal and noise
        }

        return metainformation
