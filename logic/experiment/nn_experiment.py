from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, TypeVar

import numpy as np
import tensorflow as tf
from enum import Enum
from project.logic.evaluation.task_register import TaskType
from project.logic.experiment.experiment import Experiment

T = TypeVar('T', bound='NeuralNetworkExperiment')


class NeuralNetworkExperiment(Experiment):
    """
    Abstract base class for neural network experiments that defines a common interface
    for all neural network experiment implementations.
    """

    def __init__(self, id: int, task: TaskType, model: Any, params: Dict[str, Any], parent=None,load_type="", model_file="",
                 weights_file=""):
        """
        Initialize the neural network experiment.

        Args:
            id: Experiment ID
            task: Task type (from TaskType enum)
            model: Neural network model
            params: Dictionary of parameters
            parent: Parent experiment (if any)
        """
        super().__init__(id, task, model, params, parent)

        # Common attributes for all neural network experiments
        self.history: Optional[tf.keras.callbacks.History] = None
        self.model_file_path: str = model_file
        self.weights_file_path: str = weights_file
        self.load_type: str = load_type

        # Data attributes
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # Prediction results
        self.train_predictions = None
        self.test_predictions = None
        self.train_actual = None
        self.test_actual = None

        # Training information
        self.trained_model = None
        self.train_time = 0.0
        self.is_finished = False

        # Evaluation metrics
        self.train_metrics = {}
        self.test_metrics = {}

        # Data info
        self.data_info = {}

    @abstractmethod
    def run(self) -> None:
        """Run the neural network experiment."""
        pass

    @abstractmethod
    def evaluate(self) -> None:
        """Evaluate the results of the neural network experiment."""
        pass

    @abstractmethod
    def _compile_model(self, model) -> None:
        """Compile the neural network model."""
        pass

    @abstractmethod
    def _train_model(self, model) -> Any:
        """Train the neural network model."""
        pass
    def load_model_from_file(self) -> None:
        """
        Завантажити модель з файлу.
        """
        try:
            if not self.model_file_path:
                raise ValueError("Шлях до файлу моделі не вказано")

            if self.load_type == 'Keras (.h5)':
                self.model = tf.keras.models.load_model(self.model_file_path, compile=False)
            elif self.load_type == 'TensorFlow SavedModel':
                self.model = tf.keras.models.load_model(self.model_file_path)
            elif self.load_type == 'JSON + Weights':
                with open(self.model_file_path, 'r') as json_file:
                    model_json = json_file.read()
                self.model = tf.keras.models.model_from_json(model_json)
                if self.weights_file_path:
                    self.model.load_weights(self.weights_file_path)
            else:
                raise ValueError(f"Непідтримуваний тип моделі: {self.load_type}")

        except Exception as e:
            raise ValueError(f"Помилка при завантаженні моделі: {str(e)}")
    @abstractmethod
    def _validate_data(self) -> None:
        """Validate the input data for the experiment."""
        pass

    # Common utility methods that can be implemented here
    def _convert_to_tensorflow_compatible(self, data):
        """
        Convert data to a TensorFlow-compatible format.

        Args:
            data: Input data to convert

        Returns:
            Data in TensorFlow-compatible format
        """
        if data is None:
            return None

        if isinstance(data, tf.Tensor):
            return data

        # If data is pandas DataFrame or Series
        if hasattr(data, 'values'):
            data = data.values

        # If data is iterable but not numpy array
        if not isinstance(data, np.ndarray):
            try:
                data = np.array(data)
            except Exception as e:
                raise ValueError(f"Failed to convert data of type {type(data)} to numpy array: {str(e)}")

        # Convert data to appropriate type
        if data.dtype == np.int32:
            data = data.astype(np.int32)
        elif data.dtype == np.float32:
            data = data.astype(np.float32)
        elif data.dtype == bool:
            data = data.astype(np.bool_)
        elif np.issubdtype(data.dtype, np.object_):
            # Find appropriate data type for array
            if all(isinstance(x, (int, np.integer)) for x in data.flatten() if x is not None):
                data = data.astype(np.int32)
            elif all(isinstance(x, (float, np.floating)) for x in data.flatten() if x is not None):
                data = data.astype(np.float32)
            elif all(isinstance(x, str) for x in data.flatten() if x is not None):
                pass
            else:
                try:
                    data = data.astype(np.float32)
                except Exception as e:
                    raise ValueError(f"Failed to convert array to numeric format: {str(e)}")

        try:
            tensor = tf.convert_to_tensor(data)
            return tensor
        except Exception as e:
            raise ValueError(
                f"Error converting to TensorFlow tensor: {str(e)}\nData type: {data.dtype}, Shape: {data.shape}")