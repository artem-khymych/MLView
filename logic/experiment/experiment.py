from typing import Dict
from collections.abc import Callable

from PyQt5.QtCore import pyqtSignal, QObject
from PyQt5.QtWidgets import QMessageBox
from pandas import DataFrame
from sklearn.neural_network import MLPClassifier, MLPRegressor

from project.logic.evaluation.metric_strategies.anomaly_detection_metric import AnomalyDetectionMetric
from project.logic.evaluation.metric_strategies.classification_metric import ClassificationMetric
from project.logic.evaluation.metric_strategies.clustering_metric import ClusteringMetric
from project.logic.evaluation.metric_strategies.density_estimation_metric import DensityEstimationMetric
from project.logic.evaluation.metric_strategies.dim_reduction_metric import DimReduction
from project.logic.evaluation.metric_strategies.metric_strategy import MetricStrategy, TimeSeriesMetric
from project.logic.evaluation.metric_strategies.regression_metric import RegressionMetric
from project.logic.experiment.input_data_params import InputDataParams
from project.logic.modules import task_names
import time
import pandas as pd
from sklearn.model_selection import train_test_split


class Experiment(QObject):
    experiment_finished = pyqtSignal(float)
    experiment_evaluated = pyqtSignal(object, object)
    renamed = pyqtSignal(str)

    def __init__(self, id, task, model, params, parent=None):
        super().__init__()
        self.id: int = id
        self.task: str = task
        self._name = f"Експеримент {id}"  # "Experiment {id}" -> "Експеримент {id}"
        self.description = ""

        self.model: Callable[[Dict]] = model
        self.trained_model = None
        self._params: Dict[str:any] = params
        self.input_data_params = InputDataParams()

        # Storing data in classic format
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.train_time: float = 0
        self.is_finished: bool = False
        self.metric_strategy: MetricStrategy

        # For storing results
        self.train_predictions = None
        self.test_predictions = None
        self.train_actual = None
        self.test_actual = None

        # For storing transformation results
        self.transformed_train = None
        self.transformed_test = None

        self.train_metrics = None
        self.test_metrics = None

        self.parent = parent
        self.children = []

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, new_name):
        if not isinstance(new_name, str):
            raise ValueError("Назва має бути рядком")  # "Name must be a string" -> "Назва має бути рядком"
        self._name = new_name

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, new_params):
        self._params = new_params

    def _choose_metric_strategy(self) -> MetricStrategy:
        # Get metric strategy for standard ML tasks
        if self.task == task_names.CLASSIFICATION:
            self.metric_strategy = ClassificationMetric()
        elif self.task == task_names.REGRESSION:
            self.metric_strategy = RegressionMetric()
        elif self.task == task_names.CLUSTERING:
            self.metric_strategy = ClusteringMetric()
        elif self.task == task_names.ANOMALY_DETECTION:
            self.metric_strategy = AnomalyDetectionMetric()
        elif self.task == task_names.DENSITY_ESTIMATION:
            self.metric_strategy = DensityEstimationMetric()
        elif self.task == task_names.DIMENSIONALITY_REDUCTION:
            self.metric_strategy = DimReduction()
        elif self.task == task_names.TIME_SERIES:
            self.metric_strategy = TimeSeriesMetric()

        # Get metric strategy if we have an sklearn neural network task
        if self.task == task_names.MLP:
            self.metric_strategy = self.__get_metric_strategy_for_mlp()

    def __get_metric_strategy_for_mlp(self) -> MetricStrategy:
        if isinstance(self.model, MLPClassifier):
            self.task = task_names.CLASSIFICATION
            return ClassificationMetric()
        elif self.model == MLPRegressor():
            self.task = task_names.REGRESSION
            return RegressionMetric()
        else:
            raise TypeError(f"Unrecognized model: {self.model}")

    def run(self):
        """
        Runs a machine learning experiment with the given input parameters.
        Processes data, trains the model, and evaluates results on training and test sets.
        """
        try:
            # Loading and preparing data
            self._load_data()

            # Choosing the appropriate metric for model evaluation
            self._choose_metric_strategy()

            # Training the model and measuring time
            start_time = time.time()

            # Creating a model instance with the given parameters
            model_instance = type(self.model)(**self._params)

            # Different training logic depending on the task type
            if self.task in [task_names.CLASSIFICATION, task_names.REGRESSION, task_names.MLP]:
                # Training the model
                model_instance.fit(self.X_train, self.y_train)

                # Saving results (for both training and test sets)
                self.train_predictions = model_instance.predict(self.X_train)
                self.test_predictions = model_instance.predict(self.X_test)
                self.train_actual = self.y_train
                self.test_actual = self.y_test

            elif self.task in [task_names.CLUSTERING, task_names.ANOMALY_DETECTION, task_names.DENSITY_ESTIMATION]:
                # For unsupervised tasks, target variable is not needed
                model_instance.fit(self.X_train)

                # Getting results for both sets
                if hasattr(model_instance, 'predict'):
                    self.train_predictions = model_instance.predict(self.X_train)
                    self.test_predictions = model_instance.predict(self.X_test)
                elif hasattr(model_instance, 'fit_predict') and (
                        self.task == task_names.CLUSTERING or self.task == task_names.ANOMALY_DETECTION):
                    # For some clustering algorithms
                    # For training set, use results from fit
                    self.train_predictions = model_instance.labels_ if hasattr(model_instance,
                                                                               'labels_') else model_instance.fit_predict(
                        self.X_train)
                    # For test set
                    self.test_predictions = model_instance.fit_predict(self.X_test)

            elif self.task == task_names.DIMENSIONALITY_REDUCTION:
                # Training dimensionality reduction model

                # Transforming data for both sets
                self.transformed_train = model_instance.fit_transform(self.X_train, y=None)
                self.transformed_test = model_instance.fit_transform(self.X_test, y=None)

            elif self.task == task_names.TIME_SERIES:
                # Specific processing for time series
                # Logic depends on specific implementation and model
                pass
            elif self.task == task_names.OWN_NN:
                pass

            # Saving training time
            self.train_time = time.time() - start_time

            # Saving the trained model
            self.trained_model = model_instance

            # Marking the experiment as finished
            self.is_finished = True
            self.experiment_finished.emit(self.train_time)
        except Exception as e:
            QMessageBox.warning(None, "Помилка параметрів", f"Виникла помилка у налаштованих параметрах:\n {e}")  # "Invalid parameters" -> "Помилка параметрів", "An error occurred in the configured parameters" -> "Виникла помилка у налаштованих параметрах"
        return

    def evaluate(self):
        self.train_metrics, self.test_metrics = self._calculate_metrics()
        print("Train metrics:", self.train_metrics)
        print("Test metrics:", self.test_metrics)
        self.experiment_evaluated.emit(self.train_metrics, self.test_metrics)

    def _load_data(self):
        """
        Loads data based on input data parameters.
        Processes categorical variables and splits data into X_train, X_test, y_train, y_test.
        """
        params = self.input_data_params

        if params.mode == 'single_file':
            # Loading from a single file
            data = self._load_file(params.single_file_path)

            # Data processing before splitting
            if not params.is_target_not_required():
                # For supervised learning
                X = data.drop(params.target_variable, axis=1)
                y = data[params.target_variable]

                # Splitting into training and test sets
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=params.test_percent / 100,
                    random_state=params.seed,
                    stratify=y if self.task == task_names.CLASSIFICATION else None
                )

                # Processing categorical variables for X_train and X_test
                X_combined = pd.concat([X_train, X_test])
                X_combined_encoded = self._encode_categorical_variables(X_combined)

                # Splitting back after processing
                self.X_train = X_combined_encoded.iloc[:len(X_train)]
                self.X_test = X_combined_encoded.iloc[len(X_train):]

                # Processing y if it's a categorical variable and the task is classification
                if self.task == task_names.CLASSIFICATION and y.dtype == 'object':
                    y_combined = pd.concat([y_train, y_test])
                    y_combined_encoded = self._encode_categorical_variables(pd.DataFrame(y_combined))

                    self.y_train = y_combined_encoded.iloc[:len(y_train)].values.ravel()
                    self.y_test = y_combined_encoded.iloc[len(y_train):].values.ravel()
                else:
                    self.y_train = y_train
                    self.y_test = y_test

            else:
                # For unsupervised learning
                # Splitting into training and test sets
                X_train, X_test = train_test_split(
                    data,
                    test_size=params.test_percent / 100,
                    random_state=params.seed
                )

                # Processing categorical variables for X_train and X_test
                X_combined = pd.concat([X_train, X_test])
                X_combined_encoded = self._encode_categorical_variables(X_combined)

                # Splitting back after processing
                self.X_train = X_combined_encoded.iloc[:len(X_train)]
                self.X_test = X_combined_encoded.iloc[len(X_train):]

                # For unsupervised learning, y is not needed
                self.y_train = None
                self.y_test = None
        else:
            # Loading from separate files
            X_train = self._load_file(params.x_train_file_path)
            X_test = self._load_file(params.x_test_file_path)

            # Processing categorical variables (joint processing for encoding consistency)
            X_combined = pd.concat([X_train, X_test])
            X_combined_encoded = self._encode_categorical_variables(X_combined)

            # Splitting back after processing
            self.X_train = X_combined_encoded.iloc[:len(X_train)]
            self.X_test = X_combined_encoded.iloc[len(X_train):]

            if not params.is_target_not_required():
                # For supervised learning
                y_train = self._load_file(params.y_train_file_path)
                y_test = self._load_file(params.y_test_file_path)

                # Checking if it's a DataFrame and processing accordingly
                if isinstance(y_train, pd.DataFrame):
                    if len(y_train.columns) == 1:
                        y_train = y_train.iloc[:, 0]
                    else:
                        # If multiple columns, process the first one
                        y_train = y_train.iloc[:, 0]

                if isinstance(y_test, pd.DataFrame):
                    if len(y_test.columns) == 1:
                        y_test = y_test.iloc[:, 0]
                    else:
                        # If multiple columns, process the first one
                        y_test = y_test.iloc[:, 0]

                # Processing categorical variables for y if necessary
                if self.task == task_names.CLASSIFICATION and pd.api.types.is_object_dtype(y_train):
                    y_combined = pd.concat([y_train, y_test])
                    y_combined = pd.DataFrame(y_combined, columns=['target'])
                    y_combined_encoded = self._encode_categorical_variables(y_combined)

                    self.y_train = y_combined_encoded.iloc[:len(y_train)].values.ravel()
                    self.y_test = y_combined_encoded.iloc[len(y_train):].values.ravel()
                else:
                    self.y_train = y_train
                    self.y_test = y_test
            else:
                # For unsupervised learning
                self.y_train = None
                self.y_test = None

    def _load_file(self, file_path):
        """
        Loads data from files of different formats.
        Supported formats: CSV, Excel, JSON, Parquet.

        Args:
            file_path (str): Path to the data file

        Returns:
            DataFrame: Loaded data as a pandas DataFrame
        """
        file_extension = file_path.split('.')[-1].lower()

        if file_extension == 'csv':
            return pd.read_csv(
                file_path,
                encoding=self.input_data_params.file_encoding,
                sep=self.input_data_params.file_separator
            )
        elif file_extension in ['xlsx', 'xls']:
            return pd.read_excel(file_path)
        elif file_extension == 'json':
            return pd.read_json(file_path)
        elif file_extension == 'parquet':
            return pd.read_parquet(file_path)
        else:
            raise ValueError(f"Непідтримуваний формат файлу: {file_extension}")  # "Unsupported file format" -> "Непідтримуваний формат файлу"

    def _encode_categorical_variables(self, data):
        """
        Processes categorical variables in a DataFrame according to the selected encoding method.

        Args:
            data (DataFrame): Data to process

        Returns:
            DataFrame: Data with processed categorical variables
        """
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns

        if len(categorical_columns) == 0:
            return data

        result_data = data.copy()

        # Selecting the categorical variable encoding method
        encoding_method = self.input_data_params.categorical_encoding

        if encoding_method == 'one-hot':
            # One-hot encoding (all categories as separate columns)
            for column in categorical_columns:
                # Creating one-hot encoding
                one_hot = pd.get_dummies(
                    result_data[column],
                    prefix=column,
                    drop_first=False
                )

                # Removing the original column and adding encoded columns
                result_data = pd.concat([result_data.drop(column, axis=1), one_hot], axis=1)

        elif encoding_method == 'to_categorical':
            from sklearn.preprocessing import LabelEncoder

            for column in categorical_columns:
                # Creating an encoder and training it on the data
                label_encoder = LabelEncoder()
                result_data[column] = label_encoder.fit_transform(result_data[column])

        return result_data

    def _calculate_metrics(self):
        """
        Calculates model performance metrics for training and test sets.

        Returns:
            tuple: (train_metrics, test_metrics) - tuple with metric dictionaries for each set
        """
        if not self.is_finished:
            raise BlockingIOError("The experiment isn't finished")

        train_metrics = {}
        test_metrics = {}

        if self.task in [task_names.CLASSIFICATION, task_names.REGRESSION, task_names.MLP]:
            # Metrics for training set
            train_metrics = self.metric_strategy.evaluate(self.train_actual, self.train_predictions)
            # Metrics for test set
            test_metrics = self.metric_strategy.evaluate(self.test_actual, self.test_predictions)

        elif self.task == task_names.CLUSTERING:
            # For clustering, internal metrics can be used
            train_metrics = self.metric_strategy.evaluate(self.X_train, self.train_predictions)
            test_metrics = self.metric_strategy.evaluate(self.X_test, self.test_predictions)

        elif self.task == task_names.DIMENSIONALITY_REDUCTION:
            # For dimensionality reduction
            train_metrics = self.metric_strategy.evaluate(self.X_train, self.transformed_train)
            test_metrics = self.metric_strategy.evaluate(self.X_test, self.transformed_test)

        elif self.task == task_names.ANOMALY_DETECTION:
            # For anomaly detection
            train_metrics = self.metric_strategy.evaluate(self.X_train, self.train_predictions)
            test_metrics = self.metric_strategy.evaluate(self.X_test, self.test_predictions)

        elif self.task == task_names.DENSITY_ESTIMATION:
            # For density estimation
            train_metrics = self.metric_strategy.evaluate(self.X_train, self.trained_model)
            test_metrics = self.metric_strategy.evaluate(self.X_test, self.trained_model)

        elif self.task == task_names.TIME_SERIES:
            # For time series (if train/test split is possible)
            if self.train_actual is not None and self.train_predictions is not None:
                train_metrics = self.metric_strategy.evaluate(self.train_actual, self.train_predictions)
            if self.test_actual is not None and self.test_predictions is not None:
                test_metrics = self.metric_strategy.evaluate(self.test_actual, self.test_predictions)

        # If metrics are empty, add an error message
        if not train_metrics:
            train_metrics = {"error": "Неможливо обчислити метрики для тренувального набору"}  # "Cannot calculate metrics for training set" -> "Неможливо обчислити метрики для тренувального набору"
        if not test_metrics:
            test_metrics = {"error": "Неможливо обчислити метрики для тестового набору"}  # "Cannot calculate metrics for test set" -> "Неможливо обчислити метрики для тестового набору"

        return train_metrics, test_metrics

    def get_params_for_tune(self):
        if self.input_data_params.single_file_path or (self.input_data_params.x_train_file_path and self.input_data_params.y_train_file_path):
            try:
                self._load_data()
            except Exception as e:
                QMessageBox.warning(None, "Помилка параметрів", f"Виникла помилка у налаштованих параметрах:\n {e}")  # "Invalid parameters" -> "Помилка параметрів", "An error occurred in the configured parameters" -> "Виникла помилка у налаштованих параметрах"
                return None, None
            return self.X_train, self.y_train
        else:
            return None, None