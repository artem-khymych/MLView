import numpy as np
import tensorflow as tf
from typing import Dict, Any
from project.logic.experiment.nn_experiment import NeuralNetworkExperiment
from project.logic.evaluation.task_register import TaskType, NNMetricFactory, NNModelType


class AutoencoderExperiment(NeuralNetworkExperiment):
    """
    Autoencoder experiment class with support for dimensionality reduction and anomaly detection.
    """

    def __init__(self, id: int, task, model: Any, params: Dict[str, Any], parent=None, load_type="", model_file="",
                 weights_file=""):
        super().__init__(id, task, model, params, parent, load_type=load_type, model_file=model_file,
                         weights_file=weights_file)

        self.task_spec_params = params.get('task_spec_params', {})
        self.bottleneck_layer_index = self.task_spec_params.get('bottleneck_layer_index', None)
        self.anomaly_threshold = self.task_spec_params.get('anomaly_threshold', 0.1)

        # Encoder model for obtaining latent representations
        self.encoder_model = None

        # Saved latent representations
        self.latent_train = None
        self.latent_test = None

        # Reconstructed data
        self.reconstructed_train = None
        self.reconstructed_test = None

        # Anomaly scores
        self.anomaly_scores_train = None
        self.anomaly_scores_test = None
        self.metric_strategy = NNMetricFactory().create_metric(NNModelType.AUTOENCODER, TaskType(self.task))

    def _create_encoder_model(self) -> None:
        """
        Creates an encoder model from the full autoencoder model.
        Uses bottleneck_layer_index to determine the final encoder layer.
        """
        try:
            # If bottleneck layer index is not specified, try to determine it automatically
            if self.bottleneck_layer_index is None:
                # Find the smallest layer (with fewest neurons)
                min_neurons = float('inf')
                for i, layer in enumerate(self.model.layers):
                    if hasattr(layer, 'output_shape'):
                        output_shape = layer.output_shape
                        if isinstance(output_shape, tuple) and len(output_shape) >= 2:
                            neurons = np.prod(output_shape[1:])
                            if neurons < min_neurons:
                                min_neurons = neurons
                                self.bottleneck_layer_index = i

                if self.bottleneck_layer_index is None:
                    # If not found, use the middle layer
                    self.bottleneck_layer_index = len(self.model.layers) // 2

                print(f"Automatically determined bottleneck layer: {self.bottleneck_layer_index}")

            # Create encoder model that outputs from the bottleneck layer
            bottleneck_layer = self.model.layers[self.bottleneck_layer_index]
            self.encoder_model = tf.keras.Model(
                inputs=self.model.input,
                outputs=bottleneck_layer.output
            )

            print(f"Created encoder from layer {self.bottleneck_layer_index}")
        except Exception as e:
            print(f"Error creating encoder model: {str(e)}")
            self.encoder_model = None

    def run(self) -> None:
        """
        Run the autoencoder experiment.
        """
        import time

        self._load_data()
        self._validate_data()

        try:
            self.X_train = self._convert_to_tensorflow_compatible(self.X_train)
            self.X_test = self._convert_to_tensorflow_compatible(self.X_test)
        except Exception as e:
            raise ValueError(f"Error converting data to TensorFlow format: {str(e)}")

        # Start time measurement
        start_time = time.time()

        self.load_model_from_file()
        model = self.model
        # Model compilation
        if isinstance(model, tf.keras.Model) or isinstance(model, tf.keras.Sequential):
            self._compile_model(model)

        # Model training
        if self.X_train is not None:
            try:
                # For autoencoder, target data = input data
                self.history = self._train_autoencoder(model)
            except Exception as e:
                raise ValueError(f"Error training autoencoder: {str(e)}")

        # Process results
        if self.task == TaskType.DIMENSIONALITY_REDUCTION:
            self._process_dimensionality_reduction()
        elif self.task == TaskType.ANOMALY_DETECTION:
            self._process_anomaly_detection()

        # Save trained model
        self.trained_model = model
        self.train_time = time.time() - start_time
        self.is_finished = True
        self.experiment_finished.emit(self.train_time)

    def _train_autoencoder(self, model) -> tf.keras.callbacks.History:
        """
        Train the autoencoder.
        """
        # Default training parameters
        default_fit_params = {
            'x': self.X_train,
            'y': self.X_train,  # For autoencoder, target data = input data
            'batch_size': 32,
            'epochs': 50,
            'verbose': 1,
            'validation_split': 0.2,
            'shuffle': True
        }

        # Update training parameters
        fit_params = {**default_fit_params}

        # Update with training parameters from self._params
        if 'fit_params' in self._params:
            for key, value in self._params.get('fit_params', {}).items():
                if value is not None:
                    fit_params[key] = value

        # Remove None parameters
        for key in list(fit_params.keys()):
            if fit_params[key] is None:
                del fit_params[key]

        # Train model
        history = model.fit(**fit_params)
        return history

    def _process_dimensionality_reduction(self) -> None:
        """
        Process results for dimensionality reduction task.
        """
        try:
            # Verify encoder was created
            if self.encoder_model is None:
                self._create_encoder_model()

            if self.encoder_model is None:
                raise ValueError("Failed to create encoder model")

            # Get latent representations
            self.latent_train = self.encoder_model.predict(self.X_train)
            self.latent_test = self.encoder_model.predict(self.X_test)

            # Save reconstructed data for quality assessment
            self.reconstructed_train = self.model.predict(self.X_train)
            self.reconstructed_test = self.model.predict(self.X_test)

            # Add dimensionality information
            print(f"Data successfully processed for dimensionality reduction task.")
            print(
                f"Compression: {self.data_info['original_dim']} -> {self.data_info['latent_dim']} (ratio: {self.data_info['compression_ratio']:.2f}x)")

        except Exception as e:
            print(f"Error processing data for dimensionality reduction: {str(e)}")

    def _process_anomaly_detection(self) -> None:
        """
        Process results for anomaly detection task.
        """
        try:
            # Get reconstructed data
            self.reconstructed_train = self.model.predict(self.X_train)
            self.reconstructed_test = self.model.predict(self.X_test)

            # Calculate reconstruction errors (per sample)
            reconstruction_errors_train = self._calculate_reconstruction_errors(self.X_train, self.reconstructed_train)
            reconstruction_errors_test = self._calculate_reconstruction_errors(self.X_test, self.reconstructed_test)

            # Save anomaly scores
            self.anomaly_scores_train = reconstruction_errors_train
            self.anomaly_scores_test = reconstruction_errors_test

            # Determine anomaly threshold if not set
            if self.anomaly_threshold is None:
                # Use 95th percentile of training errors
                self.anomaly_threshold = np.percentile(reconstruction_errors_train, 95)
                print(f"Automatically determined anomaly threshold: {self.anomaly_threshold:.4f}")

            # Detect anomalies
            self.train_predictions = (reconstruction_errors_train > self.anomaly_threshold).astype(int)
            self.test_predictions = (reconstruction_errors_test > self.anomaly_threshold).astype(int)

            # If labels (ground truth) exist, use them for evaluation
            if hasattr(self, 'y_train') and self.y_train is not None:
                self.train_actual = self.y_train
                self.test_actual = self.y_test
            else:
                # If no labels, assume all training data is normal
                self.train_actual = np.zeros(len(self.X_train))
                self.test_actual = np.zeros(len(self.X_test))
                print("Warning: Anomaly labels not provided. Assuming all training data is normal.")

            print(f"Data successfully processed for anomaly detection task.")

            # Add information about detected anomalies
            anomalies_train = np.sum(self.train_predictions)
            anomalies_test = np.sum(self.test_predictions)

        except Exception as e:
            print(f"Error processing data for anomaly detection: {str(e)}")

    def _calculate_reconstruction_errors(self, original: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
        """
        Calculate reconstruction errors per sample.

        Args:
            original: Original data
            reconstructed: Reconstructed data

        Returns:
            Array of reconstruction errors per sample
        """
        tf.experimental.numpy.experimental_enable_numpy_behavior()
        # Flatten data for simpler calculations
        original_flat = original.reshape(len(original), -1)
        reconstructed_flat = reconstructed.reshape(len(reconstructed), -1)

        # Calculate MSE per sample
        mse = np.mean(np.square(original_flat - reconstructed_flat), axis=1)

        return mse

    def _compile_model(self, model) -> None:
        """
        Compile the autoencoder model.
        """
        # Default compilation parameters
        default_compile_params = {
            'optimizer': 'adam',
            'loss': 'mse',
            'metrics': ['mae']
        }

        # Update default parameters with provided ones
        compile_params = {**default_compile_params}

        # Update with model parameters from self._params
        if 'model_params' in self._params:
            model_params = self._params.get('model_params', {})
            for key in ['loss', 'metrics']:
                if key in model_params and model_params[key] is not None:
                    compile_params[key] = model_params[key]

            # Handle optimizer with learning_rate
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
                    else:
                        compile_params['optimizer'] = optimizer_name
                else:
                    compile_params['optimizer'] = optimizer_name

        # Compile model
        try:
            model.compile(**compile_params)
            print(f"Autoencoder model compiled with parameters: {compile_params}")
        except Exception as e:
            raise ValueError(f"Error compiling autoencoder model: {str(e)}")

    def _validate_data(self) -> None:
        """
        Validate data compatibility with task type and model.
        """
        if self.X_train is None or self.X_test is None:
            raise ValueError("Missing input data for autoencoder")

        # Convert to numpy arrays
        if not isinstance(self.X_train, np.ndarray):
            try:
                self.X_train = np.array(self.X_train)
            except Exception as e:
                raise ValueError(f"Failed to convert X_train to numpy array: {str(e)}")

        if not isinstance(self.X_test, np.ndarray):
            try:
                self.X_test = np.array(self.X_test)
            except Exception as e:
                raise ValueError(f"Failed to convert X_test to numpy array: {str(e)}")

        # Check for NaN values
        if np.isnan(self.X_train).any():
            raise ValueError("NaN values detected in training data")
        if np.isnan(self.X_test).any():
            raise ValueError("NaN values detected in test data")

        # Check dimension compatibility
        if self.X_train.ndim != self.X_test.ndim:
            raise ValueError(f"Dimension mismatch between X_train ({self.X_train.ndim}) and X_test ({self.X_test.ndim})")

        # For anomaly detection task, check labels (if available)
        if self.task == TaskType.ANOMALY_DETECTION and hasattr(self, 'y_train') and self.y_train is not None:
            try:
                if not isinstance(self.y_train, np.ndarray):
                    self.y_train = np.array(self.y_train)
                if not isinstance(self.y_test, np.ndarray):
                    self.y_test = np.array(self.y_test)

                # Verify labels are binary (0 - normal, 1 - anomaly)
                unique_labels = np.unique(np.concatenate([self.y_train, self.y_test]))
                if not np.array_equal(unique_labels, np.array([0, 1])) and not np.array_equal(unique_labels, np.array(
                        [0])) and not np.array_equal(unique_labels, np.array([1])):
                    print(
                        f"Warning: Anomaly labels should be 0 (normal) or 1 (anomaly). Detected labels: {unique_labels}")

                    # Convert labels to binary
                    if len(unique_labels) == 2:
                        label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
                        self.y_train = np.array([label_map[label] for label in self.y_train])
                        self.y_test = np.array([label_map[label] for label in self.y_test])
                        print(f"Labels converted: {unique_labels} -> [0, 1]")

                # Save anomaly distribution information
                anomalies_train = np.sum(self.y_train)
                anomalies_test = np.sum(self.y_test)
                self.data_info['true_anomalies_train'] = anomalies_train
                self.data_info['true_anomalies_test'] = anomalies_test
                self.data_info['true_anomalies_train_percent'] = (anomalies_train / len(self.y_train)) * 100
                self.data_info['true_anomalies_test_percent'] = (anomalies_test / len(self.y_test)) * 100

                print(f"Actual anomalies in data:")
                print(
                    f"  Training data: {anomalies_train} out of {len(self.y_train)} ({self.data_info['true_anomalies_train_percent']:.2f}%)")
                print(
                    f"  Test data: {anomalies_test} out of {len(self.y_test)} ({self.data_info['true_anomalies_test_percent']:.2f}%)")

            except Exception as e:
                print(f"Error processing anomaly labels: {str(e)}")

    def evaluate(self) -> None:
        """
        Evaluate autoencoder experiment results.
        """
        if not self.is_finished:
            raise RuntimeError("Experiment must be completed before evaluation")

        # Initialize metric dictionaries
        self.train_metrics = {}
        self.test_metrics = {}

        try:
            if TaskType(self.task) == TaskType.DIMENSIONALITY_REDUCTION:
                self.train_metrics = self.metric_strategy.evaluate(self.X_train, self.reconstructed_train)
                self.test_metrics = self.metric_strategy.evaluate(self.X_test, self.reconstructed_test)
            elif TaskType(self.task) == TaskType.ANOMALY_DETECTION:
                self.train_metrics = self.metric_strategy.evaluate(self.X_train, self.train_predictions)
                self.test_metrics = self.metric_strategy.evaluate(self.X_test, self.test_predictions)
            else:
                raise ValueError(f"Unsupported task type for autoencoder: {self.task}")

        except Exception as e:
            print(f"Error evaluating results: {str(e)}")
            self.train_metrics = {"error": f"Evaluation error: {str(e)}"}
            self.test_metrics = {"error": f"Evaluation error: {str(e)}"}

        # Signal evaluation completion
        self.experiment_evaluated.emit(self.train_metrics, self.test_metrics)