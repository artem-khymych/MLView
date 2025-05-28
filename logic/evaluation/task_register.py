from enum import Enum
from dataclasses import dataclass
from typing import Set, Type

from project.logic.evaluation.metric_strategies.anomaly_detection_metric import AnomalyDetectionMetric
from project.logic.evaluation.metric_strategies.dim_reduction_metric import DimReduction


class NNModelType(Enum):
    """Types of neural networks"""
    GENERIC = "GENERIC"  # Other model types
    #CONVOLUTIONAL = "CONVOLUTIONAL"
    AUTOENCODER = "AUTOENCODER"

class TaskType(Enum):
    CLASSIFICATION = "Classification"
    REGRESSION = "Regression"
    CLUSTERING = "Clustering"
    DIMENSIONALITY_REDUCTION = "Dimensionality Reduction"
    ANOMALY_DETECTION = "Anomaly Detection"
    DENSITY_ESTIMATION = "Density Estimation"
    TIME_SERIES_FORECASTING = "Time Series Forecasting"


@dataclass
class TaskConfig:
    """Neural network task configuration"""
    task_type: TaskType  # Task type
    metric_class: Type  # Metric class for model evaluation
    description: str  # Task description


class ModelTaskRegistry:
    """Registry that stores information about task types for different neural network types"""

    def __init__(self, nn_type: NNModelType = None, taskconfig: TaskConfig = None) -> None:
        # Dictionary mapping model type to list of supported tasks
        self.taskconfig = taskconfig
        self.nn_type = nn_type
        self._model_to_tasks: Dict[NNModelType, List[TaskConfig]] = {
            model_type: [] for model_type in NNModelType
        }

        # Dictionary mapping task type to list of model types that support it
        self._task_to_models: Dict[TaskType, NNModelType] = {}
        # Initialize registry with default mappings
        self._initialize_registry()

    def _initialize_registry(self):
        """Initializes the registry with default task and model mappings"""
        # Import all required metric classes
        from project.logic.evaluation.metric_strategies.classification_metric import ClassificationMetric
        from project.logic.evaluation.metric_strategies.regression_metric import RegressionMetric
        from project.logic.evaluation.metric_strategies.metric_strategy import TimeSeriesMetric

        # Add all tasks to GENERIC model type with appropriate metrics
        self.register_task(
            NNModelType.GENERIC,
            TaskConfig(
                TaskType.CLASSIFICATION,
                ClassificationMetric,
                "Metrics for classification tasks"
            )
        )

        self.register_task(
            NNModelType.GENERIC,
            TaskConfig(
                TaskType.REGRESSION,
                RegressionMetric,
                "Metrics for regression tasks"
            )
        )

        self.register_task(
            NNModelType.AUTOENCODER,
            TaskConfig(
                TaskType.DIMENSIONALITY_REDUCTION,
                DimReduction,
                "Metrics for dimensionality reduction quality assessment"
            )
        )


        self.register_task(
            NNModelType.AUTOENCODER,
            TaskConfig(
                TaskType.ANOMALY_DETECTION,
                AnomalyDetectionMetric,
                "Metrics for anomaly detection"
            )
        )


        self.register_task(
            NNModelType.GENERIC,
            TaskConfig(
                TaskType.TIME_SERIES_FORECASTING,
                TimeSeriesMetric,
                "Metrics for time series analysis"
            )
        )

    def register_task(self, model_type: NNModelType, task_config: TaskConfig):
        if model_type not in self._model_to_tasks:
            self._model_to_tasks[model_type] = []
        self._model_to_tasks[model_type].append(task_config)

        # Assign model to task (only one)
        self._task_to_models[task_config.task_type] = model_type

    def get_tasks_for_model(self, model_type: str):
        """Gets list of tasks supported by specific model type"""
        model = NNModelType(model_type)
        return self._model_to_tasks.get(model, [])

    def get_models_for_task(self, task_type):
        """Gets list of model types that support specific task"""
        return self._task_to_models.get(task_type, NNModelType.GENERIC)

    def get_metric_class(self, model_type, task_type):
        """Gets metric class for given model type and task"""
        for task_config in self._model_to_tasks.get(model_type, []):
            if task_config.task_type == task_type:
                return task_config.metric_class
        return None


class NNMetricFactory:
    """Factory for creating appropriate metrics for different neural network tasks"""

    def __init__(self, registry: ModelTaskRegistry =None):
        self.registry = registry or ModelTaskRegistry()

    def create_metric(self, model_type: NNModelType, task_type: TaskType):
        """
        Creates appropriate metric for neural network evaluation

        Parameters:
        -----------
        model_type : NNModelType
            Neural network type
        task_type : TaskType
            Task type

        Returns:
        -----------
        MetricStrategy
            Instance of appropriate metric strategy
        """
        metric_class = self.registry.get_metric_class(model_type, task_type)

        if metric_class is None:
            raise ValueError(f"No appropriate metric found for model {model_type} and task {task_type}")

        return metric_class()


# Helper function for working with string names
def get_nn_metric(model_type_str: str, task_type_str: str):
    """
    Utility function for getting metric by string names of model type and task

    Parameters:
    -----------
    model_type_str : str
        Model type name (e.g., "GENERIC")
    task_type_str : str
        Task type name (e.g., "Classification", "Regression")

    Returns:
    -----------
    MetricStrategy
        Instance of appropriate metric strategy
    """
    # Map string values to task enums
    task_type_map = {
        "CLASSIFICATION": TaskType.CLASSIFICATION,
        "REGRESSION": TaskType.REGRESSION,
        #"CLUSTERING": TaskType.CLUSTERING,
        "DIMENSIONALITY_REDUCTION": TaskType.DIMENSIONALITY_REDUCTION,
        "ANOMALY_DETECTION": TaskType.ANOMALY_DETECTION,
        #"DENSITY_ESTIMATION": TaskType.DENSITY_ESTIMATION,
        "TIME_SERIES_FORECASTING": TaskType.TIME_SERIES_FORECASTING
    }

    try:
        model_type = NNModelType(model_type_str.upper())
    except ValueError:
        raise ValueError(f"Unknown model type: {model_type_str}")

    try:
        task_type = task_type_map[task_type_str.upper()]
    except KeyError:
        raise ValueError(f"Unknown task type: {task_type_str}")

    # Create factory and get metric
    factory = NNMetricFactory()
    return factory.create_metric(model_type, task_type)


########################################################################################################################
from enum import Enum
from typing import Dict, List, Type
from dataclasses import dataclass
from project.logic.evaluation.metric_strategies.metric_strategy import MetricStrategy


class MLTaskType(Enum):
    """Machine learning task types"""
    CLASSIFICATION = "Classification"
    REGRESSION = "Regression"
    CLUSTERING = "Clustering"
    DIMENSIONALITY_REDUCTION = "Dimensionality Reduction"
    ANOMALY_DETECTION = "Anomaly Detection"
    DENSITY_ESTIMATION = "Density Estimation"
    MLP = "Scikit-learn MLP models"
    TIME_SERIES = "Time Series"
    OWN_NN = "Import own"


@dataclass
class TaskMetricConfig:
    """Metric configuration for machine learning task"""
    metric_class: Type[MetricStrategy]  # Metric class for evaluation
    description: str  # Task and metric description


class MLTaskMetricRegistry:
    """Registry that stores information about metrics for different machine learning tasks"""

    def __init__(self, mltask_type: MLTaskType = None, task_config: TaskMetricConfig = None):
        # Dictionary mapping task type to metric configuration
        self.task_config: TaskMetricConfig = task_config
        self.mltask_type: MLTaskType = mltask_type
        self._task_to_metric: Dict[MLTaskType, TaskMetricConfig] = {}

        # Initialize registry with default mappings
        self._initialize_registry()

    def _initialize_registry(self):
        """Initializes the registry with default task and metric mappings"""
        # Import all required metric classes
        from project.logic.evaluation.metric_strategies.classification_metric import ClassificationMetric
        from project.logic.evaluation.metric_strategies.regression_metric import RegressionMetric
        from project.logic.evaluation.metric_strategies.clustering_metric import ClusteringMetric
        from project.logic.evaluation.metric_strategies.dim_reduction_metric import DimReduction
        from project.logic.evaluation.metric_strategies.anomaly_detection_metric import AnomalyDetectionMetric
        from project.logic.evaluation.metric_strategies.density_estimation_metric import DensityEstimationMetric
        from project.logic.evaluation.metric_strategies.metric_strategy import TimeSeriesMetric

        # Register metrics for each task type
        self.register_metric(
            MLTaskType.CLASSIFICATION,
            TaskMetricConfig(
                ClassificationMetric,
                "Metrics for classification tasks"
            )
        )

        self.register_metric(
            MLTaskType.REGRESSION,
            TaskMetricConfig(
                RegressionMetric,
                "Metrics for regression tasks"
            )
        )

        self.register_metric(
            MLTaskType.CLUSTERING,
            TaskMetricConfig(
                ClusteringMetric,
                "Metrics for clustering tasks"
            )
        )

        self.register_metric(
            MLTaskType.DIMENSIONALITY_REDUCTION,
            TaskMetricConfig(
                DimReduction,
                "Metrics for dimensionality reduction quality assessment"
            )
        )

        self.register_metric(
            MLTaskType.ANOMALY_DETECTION,
            TaskMetricConfig(
                AnomalyDetectionMetric,
                "Metrics for anomaly detection"
            )
        )

        self.register_metric(
            MLTaskType.DENSITY_ESTIMATION,
            TaskMetricConfig(
                DensityEstimationMetric,
                "Metrics for distribution density estimation"
            )
        )

        self.register_metric(
            MLTaskType.TIME_SERIES,
            TaskMetricConfig(
                TimeSeriesMetric,
                "Metrics for time series analysis"
            )
        )

    def register_metric(self, task_type: MLTaskType, metric_config: TaskMetricConfig):
        """Registers new metric for task type"""
        self._task_to_metric[task_type] = metric_config

    def get_metric_config(self, task_type: MLTaskType) -> TaskMetricConfig:
        """Gets metric configuration for given task type"""
        return self._task_to_metric.get(task_type)

    def get_metric_class(self, task_type: MLTaskType) -> Type[MetricStrategy]:
        """Gets metric class for given task type"""
        config = self.get_metric_config(task_type)
        return config.metric_class if config else None


class MLMetricFactory:
    """Factory for creating appropriate metrics for different machine learning tasks"""

    def __init__(self, registry=None):
        self.registry = registry or MLTaskMetricRegistry()

    def create_metric(self, task_type: MLTaskType) -> MetricStrategy:
        """
        Creates appropriate metric for machine learning model evaluation

        Parameters:
        -----------
        task_type : MLTaskType
            Machine learning task type

        Returns:
        -----------
        MetricStrategy
            Instance of appropriate metric strategy
        """
        metric_class = self.registry.get_metric_class(task_type)

        if metric_class is None:
            raise ValueError(f"No appropriate metric found for task type {task_type}")

        return metric_class()
def get_ml_metric(task_type_str: str) -> MetricStrategy:
    """
    Функція-утиліта для отримання метрики за строковою назвою типу завдання

    Parameters:
    -----------
    task_type_str : str
        Назва типу завдання (наприклад, "Classification", "Regression")

    Returns:
    -----------
    MetricStrategy
        Екземпляр відповідної стратегії метрики
    """
    # Мапимо строкові значення до enum
    task_type_map = {
        "CLASSIFICATION": MLTaskType.CLASSIFICATION,
        "REGRESSION": MLTaskType.REGRESSION,
        "CLUSTERING": MLTaskType.CLUSTERING,
        "DIMENSIONALITY_REDUCTION": MLTaskType.DIMENSIONALITY_REDUCTION,
        "ANOMALY_DETECTION": MLTaskType.ANOMALY_DETECTION,
        "DENSITY_ESTIMATION": MLTaskType.DENSITY_ESTIMATION,
        "TIME_SERIES": MLTaskType.TIME_SERIES
    }

    try:
        task_type = task_type_map[task_type_str.upper()]
    except KeyError:
        raise ValueError(f"Невідомий тип завдання: {task_type_str}")

    # Створюємо фабрику та отримуємо метрику
    factory = MLMetricFactory()
    return factory.create_metric(task_type)
