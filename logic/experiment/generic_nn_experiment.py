import time
from typing import Dict, Any, Optional
import tensorflow as tf
from PyQt5.QtWidgets import QMessageBox
import numpy as np

from project.logic.experiment.experiment import Experiment
from project.logic.experiment.nn_experiment import NeuralNetworkExperiment
from project.logic.experiment.nn_input_data_params import NeuralNetInputDataParams
from project.logic.evaluation.task_register import TaskType, NNMetricFactory, get_nn_metric, \
    NNModelType


class GenericNeuralNetworkExperiment(NeuralNetworkExperiment):
    """
    Neural network experiment class with support for different architectures
    and saving specific hyperparameters for each task type.
    """

    def __init__(self, id: int, task, model: Any, params: Dict[str, Any], parent=None, load_type="", model_file="",
                 weights_file=""):
        super().__init__(id, task, model, params, parent, load_type=load_type, model_file=model_file,
                         weights_file=weights_file)

        # Training history
        self.history = None
        # Model file paths
        self.input_data_params: NeuralNetInputDataParams = NeuralNetInputDataParams()

        # Model task registry
        self.metric_factory = NNMetricFactory()

        # For storing data information
        self.data_info = {}

    def get_params_for_tune(self):
        self._load_data()

        self._validate_data()
        try:
            self.X_train = self._convert_to_tensorflow_compatible(self.X_train)
            self.X_test = self._convert_to_tensorflow_compatible(self.X_test)
            if self.y_train is not None:
                self.y_train = self._convert_to_tensorflow_compatible(self.y_train)
            if self.y_test is not None:
                self.y_test = self._convert_to_tensorflow_compatible(self.y_test)
        except Exception as e:
            raise ValueError(f"Помилка при перетворенні даних в формат TensorFlow: {str(e)}")

        return self.X_train, self.y_train

    def run(self) -> None:
        """
        Run neural network experiment.
        """
        try:
            self._load_data()

            self._validate_data()
            try:
                self.X_train = self._convert_to_tensorflow_compatible(self.X_train)
                self.X_test = self._convert_to_tensorflow_compatible(self.X_test)
                if self.y_train is not None:
                    self.y_train = self._convert_to_tensorflow_compatible(self.y_train)
                if self.y_test is not None:
                    self.y_test = self._convert_to_tensorflow_compatible(self.y_test)
            except Exception as e:
                raise ValueError(f"Помилка при перетворенні даних в формат TensorFlow: {str(e)}")

            # Start time measurement
            start_time = time.time()

            if self.model_file_path:
                # If model is loaded from file
                self.load_model_from_file()
            model = self.model

            # Compile model (if it's a Keras model)
            if isinstance(model, tf.keras.Model) or isinstance(model, tf.keras.Sequential):
                self._compile_model(model)

            # Train model (if needed)
            if self.X_train is not None:
                try:
                    self.history = self._train_model(model)
                except Exception as e:
                    error_msg = str(e)
                    if "SparseSoftmaxCrossEntropyWithLogits" in error_msg and "valid range" in error_msg:
                        raise ValueError(
                            f"Помилка при навчанні: значення міток виходять за межі допустимого діапазону.\n"
                            f"Перевірте формат міток та їх відповідність кількості класів у моделі.\n"
                            f"Деталі: {error_msg}"
                        )
                    else:
                        raise

            # Predictions
            if self.task == TaskType.REGRESSION:
                self.train_predictions = model.predict(self.X_train)
                self.test_predictions = model.predict(self.X_test)
                self.train_actual = self.y_train
                self.test_actual = self.y_test
            elif self.task == TaskType.CLASSIFICATION:
                train_probabilities = model.predict(self.X_train)
                test_probabilities = model.predict(self.X_test)

                self.train_predictions = self._convert_probabilities_to_classes(train_probabilities)
                self.test_predictions = self._convert_probabilities_to_classes(test_probabilities)

                self.train_actual = self.y_train
                self.test_actual = self.y_test
            elif self.task == TaskType.TIME_SERIES_FORECASTING:
                # Time series require special processing
                self._process_time_series_predictions(model)
            # Save trained model
            self.trained_model = model
            self.train_time = time.time() - start_time
            self.is_finished = True
            self.experiment_finished.emit(self.train_time)
        except Exception as e:
            QMessageBox.warning(None, "Хибні параметри", f"Винила помилка у налаштованих параметрах:\n {e}")
        return

    def _process_time_series_predictions(self, model):

        try:
            x_train = self._convert_to_tensorflow_compatible(self.X_train)
            x_test = self._convert_to_tensorflow_compatible(self.X_test)

            if isinstance(x_train, (np.ndarray, tf.Tensor)) and len(x_train.shape) == 3:
                self.train_predictions = model.predict(x_train)
                self.test_predictions = model.predict(x_test)
                self.train_actual = self.y_train
                self.test_actual = self.y_test
            else:
                raise NotImplementedError("Processing for this data type is not implemented")
        except Exception as e:
            print(f"Error processing time series: {str(e)}")

    def _validate_data(self):
        """
        Check data correspondence to task type and model.
        """
        if self.task == TaskType.CLASSIFICATION:
            # Validation for classification tasks
            self._validate_classification_data()
        elif self.task == TaskType.REGRESSION:
            # Validation for regression tasks
            self._validate_regression_data()
        elif self.task == TaskType.TIME_SERIES_FORECASTING:
            # Validation for time series forecasting tasks
            self._validate_time_series_data()

        # Store data information
        self._store_data_info()

    def _validate_classification_data(self):
        """
        Validate data for classification task.
        """
        if self.y_train is None or self.y_test is None:
            raise ValueError("Class labels are required for classification")

        # Check that labels are integers for sparse_categorical_crossentropy
        if not isinstance(self.y_train, np.ndarray):
            self.y_train = np.array(self.y_train)
        if not isinstance(self.y_test, np.ndarray):
            self.y_test = np.array(self.y_test)

        # Label validation
        unique_labels = np.unique(np.concatenate([self.y_train, self.y_test]))
        num_classes = len(unique_labels)

        if num_classes == 2:
            # Binary classification
            if np.min(unique_labels) < 0 or np.max(unique_labels) > 1:
                # Transform labels for binary classification (0 and 1)
                self._transform_binary_labels()
        elif num_classes > 2:
            # Multi-class classification
            if np.min(unique_labels) < 0:
                raise ValueError(f"Negative label values detected: {unique_labels}")

            # Check for sequential labels (starting from 0)
            if not np.array_equal(unique_labels, np.arange(num_classes)):
                # Transform labels for sequential numbering from 0
                self._transform_multiclass_labels()
        else:
            raise ValueError(f"Too few classes in data: {num_classes}")

    def _transform_binary_labels(self):
        """
        Transform labels for binary classification.
        """
        unique_labels = np.unique(np.concatenate([self.y_train, self.y_test]))
        print(f"Transforming binary classification labels: {unique_labels} -> [0, 1]")

        # Transform to 0 and 1
        label_map = {unique_labels[0]: 0, unique_labels[1]: 1}

        # Store for use in evaluation
        self.data_info['original_labels'] = unique_labels
        self.data_info['label_map'] = label_map

        # Transform labels
        self.y_train = np.array([label_map[label] for label in self.y_train])
        self.y_test = np.array([label_map[label] for label in self.y_test])

    def _transform_multiclass_labels(self):
        """
        Transform labels for multi-class classification.
        """
        unique_labels = np.unique(np.concatenate([self.y_train, self.y_test]))
        print(f"Transforming multi-class classification labels: {unique_labels} -> [0...{len(unique_labels) - 1}]")

        # Create label mapping
        label_map = {label: i for i, label in enumerate(unique_labels)}

        # Store for use in evaluation
        self.data_info['original_labels'] = unique_labels
        self.data_info['label_map'] = label_map
        self.data_info['num_classes'] = len(unique_labels)

        # Transform labels
        self.y_train = np.array([label_map[label] for label in self.y_train])
        self.y_test = np.array([label_map[label] for label in self.y_test])

    def _validate_regression_data(self):
        """
        Validate data for regression task.
        """
        if self.y_train is None or self.y_test is None:
            raise ValueError("Target values are required for regression")

        # Convert to numpy arrays
        if not isinstance(self.y_train, np.ndarray):
            self.y_train = np.array(self.y_train)
        if not isinstance(self.y_test, np.ndarray):
            self.y_test = np.array(self.y_test)

        # Check for invalid values
        if np.isnan(self.y_train).any() or np.isnan(self.y_test).any():
            raise ValueError("NaN values detected in target variables")

        # Store data information
        self.data_info['y_min'] = min(np.min(self.y_train), np.min(self.y_test))
        self.data_info['y_max'] = max(np.max(self.y_train), np.max(self.y_test))

    def _validate_time_series_data(self):
        """
        Validate data for time series forecasting task.
        """
        # Basic validation
        if self.X_train is None or self.X_test is None:
            raise ValueError("Time series data is missing")

        # Check data shape
        if isinstance(self.X_train, np.ndarray):
            if len(self.X_train.shape) != 3:
                raise ValueError(
                    f"Incorrect input data shape for time series. Expected 3D array, got: {self.X_train.shape}")
        elif isinstance(self.X_train, tf.keras.utils.Sequence):
            # For sequence generators, validation is performed during loading
            pass
        else:
            raise ValueError(f"Unsupported data type for time series: {type(self.X_train)}")

    def _store_data_info(self):
        """
        Store data information for use in other methods.
        """
        # Basic data information
        try:
            if isinstance(self.X_train, np.ndarray):
                self.data_info['x_shape'] = self.X_train.shape
            if isinstance(self.y_train, np.ndarray):
                self.data_info['y_shape'] = self.y_train.shape

            if self.task == TaskType.CLASSIFICATION:
                unique_train = np.unique(self.y_train)
                unique_test = np.unique(self.y_test)
                all_unique = np.unique(np.concatenate([unique_train, unique_test]))

                self.data_info['num_classes'] = len(all_unique)
                self.data_info['unique_labels'] = all_unique
                print(f"Detected {self.data_info['num_classes']} classes with labels: {all_unique}")

                # Check for missing classes in training data
                if len(unique_train) != len(all_unique):
                    print(f"Warning: Some classes are missing in training data. Training labels: {unique_train}")

            elif self.task == TaskType.REGRESSION:
                # Additional information for regression
                if isinstance(self.y_train, np.ndarray) and isinstance(self.y_test, np.ndarray):
                    self.data_info['y_mean'] = np.mean(np.concatenate([self.y_train, self.y_test]))
                    self.data_info['y_std'] = np.std(np.concatenate([self.y_train, self.y_test]))

        except Exception as e:
            print(f"Error collecting data information: {str(e)}")

    def _compile_model(self, model) -> None:
        """
        Compile Keras model with task-specific parameters.

        Args:
            model: Keras model to compile
        """
        # Default compilation parameters
        default_compile_params = {
            'optimizer': 'adam',
            'metrics': ['accuracy']
        }

        # Determine loss function based on task type and data characteristics
        if 'loss' not in self._params:
            if self.task == TaskType.CLASSIFICATION:
                # For classification, choice depends on number of classes and label format
                num_classes = self.data_info.get('num_classes', 0)

                # Check model's output layer
                try:
                    output_units = model.layers[-1].output_shape[-1]
                    if output_units != num_classes:
                        print(f"Warning: Number of neurons in output layer ({output_units}) "
                              f"doesn't match number of classes ({num_classes})")
                except:
                    output_units = None

                if num_classes == 2:
                    # For binary classification
                    default_compile_params['loss'] = 'binary_crossentropy'
                    # Ensure last layer has sigmoid activation
                    try:
                        last_layer_activation = model.layers[-1].activation.__name__
                        if last_layer_activation != 'sigmoid':
                            print(f"Warning: For binary classification, sigmoid activation is recommended "
                                  f"in output layer, not {last_layer_activation}")
                    except:
                        pass
                else:
                    # For multi-class classification
                    default_compile_params['loss'] = 'sparse_categorical_crossentropy'
                    # Ensure last layer has softmax activation
                    try:
                        last_layer_activation = model.layers[-1].activation.__name__
                        if last_layer_activation != 'softmax':
                            print(f"Warning: For multi-class classification, softmax activation is recommended "
                                  f"in output layer, not {last_layer_activation}")
                    except:
                        pass
            elif self.task == TaskType.REGRESSION:
                default_compile_params['loss'] = 'mse'
                default_compile_params['metrics'] = ['mae', 'mse']
            elif self.task == TaskType.ANOMALY_DETECTION:
                default_compile_params['loss'] = 'binary_crossentropy'
            elif self.task == TaskType.TIME_SERIES_FORECASTING:
                default_compile_params['loss'] = 'mse'
                default_compile_params['metrics'] = ['mae']
            else:
                default_compile_params['loss'] = 'mse'  # Default

        # Update default parameters with passed parameters
        compile_params = {**default_compile_params}

        # Update with model parameters in self._params
        if 'model_params' in self._params:
            model_params = self._params.get('model_params', {})
            for key in ['loss', 'metrics']:
                if key in model_params and model_params[key] is not None:
                    compile_params[key] = model_params[key]

            # Handle optimizer with learning_rate consideration
            if 'optimizer' in model_params and model_params['optimizer'] is not None:
                optimizer_name = model_params['optimizer']
                learning_rate = model_params.get('learning_rate')

                # Create optimizer object with specified learning_rate
                if learning_rate is not None:
                    if optimizer_name == 'adam':
                        compile_params['optimizer'] = tf.keras.optimizers.Adam(learning_rate=learning_rate)
                    elif optimizer_name == 'sgd':
                        compile_params['optimizer'] = tf.keras.optimizers.SGD(learning_rate=learning_rate)
                    elif optimizer_name == 'rmsprop':
                        compile_params['optimizer'] = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
                    elif optimizer_name == 'adagrad':
                        compile_params['optimizer'] = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
                    elif optimizer_name == 'adadelta':
                        compile_params['optimizer'] = tf.keras.optimizers.Adadelta(learning_rate=learning_rate)
                    elif optimizer_name == 'adamax':
                        compile_params['optimizer'] = tf.keras.optimizers.Adamax(learning_rate=learning_rate)
                    elif optimizer_name == 'nadam':
                        compile_params['optimizer'] = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
                    else:
                        # If optimizer name is not recognized, use simple string name
                        compile_params['optimizer'] = optimizer_name
                        print(
                            f"Warning: Unrecognized optimizer '{optimizer_name}'. Learning rate will be ignored.")
                else:
                    # If learning_rate is not specified, use just the optimizer name
                    compile_params['optimizer'] = optimizer_name

        # Compile model
        try:
            model.compile(**compile_params)
            print(f"Model compiled with parameters: {compile_params}")
        except Exception as e:
            raise ValueError(f"Error compiling model: {str(e)}")

    def _train_model(self, model):
        """
        Train Keras model with task-specific parameters.

        Args:
            model: Keras model to train

        Returns:
            History: Training history
        """
        # Default training parameters
        default_fit_params = {
            'x': self.X_train,
            'batch_size': 32,
            'epochs': 10,
            'verbose': 1,
            'validation_split': 0.2,
            'shuffle': True,
            'y': self.y_train
        }

        # Add target variables depending on task type
        if self.task in [TaskType.CLASSIFICATION, TaskType.REGRESSION]:
            default_fit_params['y'] = self.y_train

        # For tasks with generators
        if isinstance(self.X_train, tf.keras.utils.Sequence):
            # For generators, no need to specify y
            default_fit_params = {
                'x': self.X_train,
                'batch_size': None,  # Generator determines batch size itself
                'epochs': 10,
                'verbose': 1,
                'validation_data': self.X_test,
                'shuffle': False  # Generator can shuffle data itself
            }
            if 'y' in default_fit_params:
                del default_fit_params['y']
            if 'validation_split' in default_fit_params:
                del default_fit_params['validation_split']

        # Update training parameters
        fit_params = {**default_fit_params}

        # Update with training parameters in self._params
        if 'fit_params' in self._params:
            for key, value in self._params.get('fit_params', {}).items():
                if value is not None:
                    fit_params[key] = value

        # Remove None parameters
        for key in list(fit_params.keys()):
            if fit_params[key] is None:
                del fit_params[key]

        # Check for empty class_weight dictionary
        if 'class_weight' in fit_params and fit_params['class_weight'] == {}:
            del fit_params['class_weight']

        # Add balanced class weights for imbalanced classification data
        if self.task == TaskType.CLASSIFICATION and 'class_weight' not in fit_params:
            try:
                unique_train_labels, counts = np.unique(self.y_train, return_counts=True)
                if len(unique_train_labels) > 1 and np.max(counts) / np.min(counts) > 5:
                    print("Detected imbalanced dataset. Applying automatic class weights.")
                    class_weights = {}
                    for i, count in enumerate(counts):
                        class_weights[i] = len(self.y_train) / (len(unique_train_labels) * count)
                    fit_params['class_weight'] = class_weights
            except Exception as e:
                print(f"Failed to automatically determine class weights: {str(e)}")

        if 'x' in fit_params and not isinstance(fit_params['x'], tf.keras.utils.Sequence):
            fit_params['x'] = self._convert_to_tensorflow_compatible(fit_params['x'])
        if 'y' in fit_params and not isinstance(fit_params['y'], tf.keras.utils.Sequence):
            fit_params['y'] = self._convert_to_tensorflow_compatible(fit_params['y'])
        if 'validation_data' in fit_params and not isinstance(fit_params['validation_data'], tf.keras.utils.Sequence):
            if isinstance(fit_params['validation_data'], tuple):
                # If validation_data is a tuple (x_val, y_val)
                x_val, y_val = fit_params['validation_data']
                x_val = self._convert_to_tensorflow_compatible(x_val)
                y_val = self._convert_to_tensorflow_compatible(y_val)
                fit_params['validation_data'] = (x_val, y_val)
            else:
                fit_params['validation_data'] = self._convert_to_tensorflow_compatible(fit_params['validation_data'])

        # Train model
        try:
            history = model.fit(**fit_params)
            return history
        except Exception as e:
            error_msg = str(e)
            if "SparseSoftmaxCrossEntropyWithLogits" in error_msg and "valid range" in error_msg:
                # Detailed error message about class labels
                unique_labels = np.unique(self.y_train)

                error_details = (
                    f"Помилка при навчанні моделі класифікації:\n"
                    f"Виявлено мітки: {unique_labels}\n"
                    # f"Оригінальна помилка: {error_msg}"
                )
                raise ValueError(error_details)
            else:
                raise

    def evaluate(self) -> None:
        """
        Simplified evaluation of results - gets metric strategy for given network and task,
        and calls it with appropriate parameters
        """
        if not self.is_finished:
            raise RuntimeError("Experiment must be completed before evaluation")

        # Initialize metric dictionaries
        self.train_metrics = {}
        self.test_metrics = {}

        try:
            # Get metric strategy for given model and task combination
            self.metric_strategy = get_nn_metric(NNModelType.GENERIC.value, self.task.value)

            # Determine input data for evaluation depending on task type
            if self.task in [TaskType.CLASSIFICATION, TaskType.REGRESSION, TaskType.TIME_SERIES_FORECASTING]:
                # For classification, regression and forecasting - use actual and predictions
                train_input = (self.train_actual, self.train_predictions)
                test_input = (self.test_actual, self.test_predictions)
            else:
                # Universal approach for other tasks
                train_input = (self.X_train, self.train_predictions) if hasattr(self, 'train_predictions') else None
                test_input = (self.X_test, self.test_predictions) if hasattr(self, 'test_predictions') else None

            # Perform evaluation for training data if available
            if train_input and all(x is not None for x in train_input):
                self.train_metrics.update(
                    self.metric_strategy.evaluate(*train_input)
                )

            # Perform evaluation for test data if available
            if test_input and all(x is not None for x in test_input):
                self.test_metrics.update(
                    self.metric_strategy.evaluate(*test_input)
                )

        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            self.train_metrics = {"error": f"Evaluation error: {str(e)}"}
            self.test_metrics = {"error": f"Evaluation error: {str(e)}"}

        # Signal evaluation completion
        self.experiment_evaluated.emit(self.train_metrics, self.test_metrics)

    def _convert_probabilities_to_classes(self, predictions, threshold=0.5):
        if predictions is None:
            return None

        # Check prediction shape
        if not isinstance(predictions, np.ndarray):
            try:
                predictions = np.array(predictions)
            except Exception as e:
                raise ValueError(f"{str(e)}")

        # For binary classification with single output (sigmoid)
        if len(predictions.shape) == 1 or (len(predictions.shape) == 2 and predictions.shape[1] == 1):
            return (predictions > threshold).astype(int)

        # For multi-class classification (softmax)
        elif len(predictions.shape) == 2 and predictions.shape[1] > 1:
            return np.argmax(predictions, axis=1)

        # For other prediction formats
        else:
            raise ValueError(f"{predictions.shape}")

    def _load_data(self):
        """
        Load data depending on neural network type and task.
        Extended method that handles different data types: tabular, images,
        sequences, text and specialized data for autoencoders.
        """
        super()._load_data()
        print(
            f"Data loaded for generic model: X_train={self.X_train.shape if hasattr(self.X_train, 'shape') else 'N/A'}")

        # Check data after loading
        self._check_loaded_data()

    def _check_loaded_data(self):
        """
        Check correctness of loaded data before use.
        """
        # Check data availability
        if self.X_train is None:
            raise ValueError("Training data not loaded (X_train is None)")
        # Check X and y correspondence
        if self.task in [TaskType.CLASSIFICATION, TaskType.REGRESSION]:
            if self.y_train is None:
                raise ValueError("Training target variables not loaded (y_train is None)")

            if isinstance(self.X_train, np.ndarray) and isinstance(self.y_train, np.ndarray):
                if len(self.X_train) != len(self.y_train):
                    raise ValueError(
                        f"X_train ({len(self.X_train)}) and y_train ({len(self.y_train)}) dimensions don't match")

                if len(self.X_test) != len(self.y_test):
                    raise ValueError(
                        f"X_test ({len(self.X_test)}) and y_test ({len(self.y_test)}) dimensions don't match")

    def _convert_to_tensorflow_compatible(self, data):

        if data is None:
            return None

        if isinstance(data, tf.Tensor):
            return data

        # If data is pandas DataFrame or Series
        if hasattr(data, 'values'):
            data = data.values

        # If data is iterable object
        if not isinstance(data, np.ndarray):
            try:
                data = np.array(data)
            except Exception as e:
                raise ValueError(f"Failed to convert data type {type(data)} to numpy array: {str(e)}")

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