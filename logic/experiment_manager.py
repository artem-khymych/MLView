import inspect
import copy
import os
import pickle
from typing import Optional, Tuple

from PyQt5.QtCore import pyqtSignal, QObject, Qt
from PyQt5.QtWidgets import QMessageBox, QFileDialog, QPushButton, QLabel, QMainWindow, QWidget, QVBoxLayout, QComboBox
from sklearn.base import ClassifierMixin, RegressorMixin, is_regressor, is_classifier

from .evaluation.task_register import ModelTaskRegistry, NNModelType, TaskType
from .experiment.autoencoder_experiment import AutoencoderExperiment
from .experiment.experiment import Experiment
from .experiment.generic_nn_experiment import GenericNeuralNetworkExperiment
from .modules import models_manager, task_names
from ..ui.experiment_settings_dialog.experiment_comparison_dialog import ExperimentComparisonDialog
from ..ui.node import Node
from tensorflow.keras import layers, Model


class ExperimentManager(QObject):
    _instance = None  # Змінна класу для зберігання екземпляра
    get_all_task_experiments = pyqtSignal(object)
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        super().__init__()
        self.experiments = {}
        self.current_node = None
        self.current_model = None
        self.current_params = {}
        self.current_task = None

    def get_node(self, node: Node):
        self.current_node = node
        self._check_experiment_data()

    def update_name(self, id, name):
        self.experiments[id].name = name

    def get_ml_model(self, task, model, params):
        self.current_model = model
        self.current_params = params
        self.current_task = task
        self._check_experiment_data()

    def _check_experiment_data(self):
        if self.current_node is not None and self.current_model is not None and self.current_params is not None and self.current_task is not None:
            self.create_new_experiment()

    def create_nn_experiment(self, task, model_file_path, weights_file_path, load_type):
        self.current_params = {}
        model_params = {
            'optimizer': 'adam',  # Оптимізатор
            'learning_rate': 0.001,
            'loss': 'sparse_categorical_crossentropy',  # Функція втрат
            'metrics': ['accuracy'],  # Метрики
            'initial_epoch': 0,  # Початковий етап
            'trainable': True,  # Чи тренувати модель
            'activation': 'relu',  # Функція активації
        }

        # Параметри методу fit():
        fit_params = {
            'batch_size': 32,  # Розмір пакету
            'epochs': 10,  # Кількість етапів
            'callbacks': [],  # Колбеки для додаткової функціональності
            'validation_split': 0.2,  # Частина для валідації
            'shuffle': True,  # Перемішувати дані
            'initial_epoch': 0,  # Початковий етап
            'class_weight': {},  # Вага класів
        }

        self.current_params["model_params"] = model_params
        self.current_params["fit_params"] = fit_params
        model = ModelTaskRegistry().get_models_for_task(task_type=TaskType(task))

        if model == NNModelType.GENERIC:
            experiment = GenericNeuralNetworkExperiment(self.current_node.id, task, Model(), self.current_params,
                                                        load_type=load_type, weights_file=weights_file_path,
                                                        model_file=model_file_path)

        elif model == NNModelType.AUTOENCODER:
            task_spec_params = {
                "bottleneck_layer_index":1
            }
            if task == TaskType.ANOMALY_DETECTION.value:
                task_spec_params["threshold"] = 0.1

            self.current_params["task_spec_params"] = task_spec_params
            experiment = AutoencoderExperiment(self.current_node.id, task, Model(), self.current_params,
                                               load_type=load_type, weights_file=weights_file_path,
                                               model_file=model_file_path)

        self.experiments[self.current_node.id] = experiment
        print(f"Created new experiment with ID: {self.current_node.id}")

        self.current_node = None
        self.current_model = None
        self.current_params = None
        self.current_task = None

        return experiment

    def create_new_experiment(self):
        experiment = Experiment(self.current_node.id, self.current_task, self.current_model, self.current_params)
        self.experiments[self.current_node.id] = experiment
        print(f"Created new experiment with ID: {self.current_node.id}")

        self.current_node = None
        self.current_model = None
        self.current_params = None
        self.current_task = None

        return experiment

    def inherit_experiment_from(self, parent_id, child_id):
        if parent_id not in self.experiments:
            print(f"Error: Parent experiment with ID {parent_id} not found")
            return None

        parent_experiment = self.experiments[parent_id]

        child_experiment = type(parent_experiment)(
            id=child_id,
            task=parent_experiment.task,
            model=parent_experiment.model,
            params=copy.deepcopy(parent_experiment._params),
            parent=parent_experiment)
        child_experiment.__dict__.update(parent_experiment.__dict__)
        child_experiment.id = child_id
        child_experiment.parent = parent_experiment
        child_experiment.input_data_params = copy.deepcopy(parent_experiment.input_data_params)
        child_experiment.description = f"Успадковано від '{parent_experiment.name}'"
        child_experiment._name = f"Успадкований {parent_experiment.name}"
        child_experiment.is_finished = False
        if hasattr(child_experiment, "history"):
            child_experiment.history = None
        self.experiments[child_id] = child_experiment

        print(f"Inherited experiment created with ID: {child_id} from parent ID: {parent_id}")
        return child_experiment

    def get_experiment(self, experiment_id):
        if experiment_id in self.experiments:
            return self.experiments[experiment_id]
        return None

    def get_related_experiments(self, experiment_id):

        if experiment_id not in self.experiments:
            print(f"Experiment with ID {experiment_id} not found.")
            return []

        related_experiments = []
        main_experiment = self.experiments[experiment_id]

        if main_experiment.is_finished:
            related_experiments.append(main_experiment)

        parent = main_experiment.parent
        while parent is not None:
            if parent.is_finished:
                related_experiments.append(parent)
            parent = parent.parent

        def add_children(experiment):
            for child in experiment.children:
                if child.is_finished:
                    related_experiments.append(child)
                add_children(child)

        add_children(main_experiment)

        if main_experiment.parent:
            for sibling in main_experiment.parent.children:
                if sibling.id != experiment_id and sibling.is_finished:
                    related_experiments.append(sibling)
                    add_children(sibling)

        return related_experiments

    def show_comparison_dialog(self, experiment_id):

        related_experiments = self.get_related_experiments(experiment_id)

        if not related_experiments:
            print(f"No completed experiments related to ID {experiment_id} found.")
            return

        dialog = ExperimentComparisonDialog(related_experiments)
        dialog.exec_()

    def _save_ml_model(self, model, parent_widget):

        file_path, _ = QFileDialog.getSaveFileName(
            parent_widget,
            "Зберегти модель",
            "",
            "Pickle Files (*.pkl);;All Files (*)"
        )

        if file_path:
            try:
                if not file_path.endswith('.pkl'):
                    file_path += '.pkl'

                # Зберігаємо модель
                with open(file_path, 'wb') as f:
                    pickle.dump(model, f)

                QMessageBox.information(
                    parent_widget,
                    "Успіх",
                    f"Модель успішно збережено у файл:\n{file_path}"
                )

                return file_path
            except Exception as e:
                QMessageBox.critical(
                    parent_widget,
                    "Помилка",
                    f"Не вдалося зберегти модель. Помилка:\n{str(e)}"
                )
                return None
        else:
            return None

    def _save_nn_model(self, experiment, parent_widget):

        def save_keras_h5(model, parent_widget, experiment) -> Optional[Tuple[str, str]]:
            default_path = ""
            if experiment.model_file_path and experiment.model_file_path.endswith('.h5'):
                default_path = experiment.model_file_path

            file_path, _ = QFileDialog.getSaveFileName(
                parent_widget,
                "Зберегти модель Keras",
                default_path,
                "Keras Model (*.h5)"
            )

            if not file_path:
                return None

            if not file_path.endswith('.h5'):
                file_path += '.h5'

            model.save(file_path)
            return file_path, ""

        def save_tf_savedmodel(model, parent_widget, experiment) -> Optional[Tuple[str, str]]:
            # Визначаємо запропоновану директорію для збереження
            default_dir = ""
            if experiment.model_file_path:
                default_dir = os.path.dirname(experiment.model_file_path)

            dir_path = QFileDialog.getExistingDirectory(
                parent_widget,
                "Виберіть директорію для збереження моделі TensorFlow",
                default_dir
            )

            if not dir_path:
                return None

            model.save(dir_path, save_format='tf')
            return dir_path, ""

        def save_json_weights(model, parent_widget, experiment) -> Optional[Tuple[str, str]]:
            default_name = ""
            if experiment.model_file_path:
                default_dir = os.path.dirname(experiment.model_file_path)
                default_name = os.path.splitext(os.path.basename(experiment.model_file_path))[0]
            else:
                default_dir = ""

            json_path, _ = QFileDialog.getSaveFileName(
                parent_widget,
                "Зберегти структуру моделі (JSON)",
                os.path.join(default_dir, default_name) if default_name else "",
                "JSON Files (*.json)"
            )

            if not json_path:
                return None

            if not json_path.endswith('.json'):
                json_path += '.json'

            base_path = os.path.splitext(json_path)[0]
            weights_path = base_path + ".weights.h5"

            json_config = model.to_json()
            with open(json_path, 'w') as json_file:
                json_file.write(json_config)

            model.save_weights(weights_path)

            return json_path, weights_path

        def perform_save(save_format):
            try:
                model = experiment.model
                result = None

                if save_format == "Keras (.h5)":
                    result = save_keras_h5(model, parent_widget, experiment)
                elif save_format == "TensorFlow SavedModel":
                    result = save_tf_savedmodel(model, parent_widget, experiment)
                elif save_format == "JSON + Weights":
                    result = save_json_weights(model, parent_widget, experiment)

                if result:
                    model_path, weights_path = result
                    experiment.model_file_path = model_path
                    if weights_path:
                        experiment.weights_file_path = weights_path
                    experiment.load_type = save_format

                    QMessageBox.information(
                        parent_widget,
                        "Успіх",
                        f"Модель успішно збережено у форматі {save_format}"
                    )

                if save_window:
                    save_window.close()

                return result
            except Exception as e:
                QMessageBox.critical(
                    parent_widget,
                    "Помилка",
                    f"Не вдалося зберегти модель. Помилка:\n{str(e)}"
                )
                return None

        save_window = None
        if parent_widget is None:
            save_window = QMainWindow()
            save_window.setWindowTitle("Збереження моделі нейронної мережі")
            save_window.resize(500, 300)

            central_widget = QWidget()
            save_window.setCentralWidget(central_widget)
            layout = QVBoxLayout(central_widget)

            info_label = QLabel("Виберіть формат і місце збереження моделі нейронної мережі")
            info_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(info_label)

            format_label = QLabel("Формат збереження:")
            layout.addWidget(format_label)

            format_combo = QComboBox()
            format_combo.addItems(["Keras (.h5)", "TensorFlow SavedModel", "JSON + Weights"])
            if experiment.load_type and experiment.load_type in ["Keras (.h5)", "TensorFlow SavedModel",
                                                                 "JSON + Weights"]:
                format_combo.setCurrentText(experiment.load_type)
            layout.addWidget(format_combo)

            save_button = QPushButton("Зберегти модель")
            layout.addWidget(save_button)

            # Показуємо вікно
            save_window.show()
            parent_widget = save_window

            def on_save_button_clicked():
                save_format = format_combo.currentText()
                perform_save(save_format)

            save_button.clicked.connect(on_save_button_clicked)

        else:
            formats = ["Keras (.h5)", "TensorFlow SavedModel", "JSON + Weights"]
            selected_format = experiment.load_type if experiment.load_type in formats else formats[0]

            perform_save(selected_format)

    def save_model(self, experiment, parent_widget=None):
        if isinstance(experiment, GenericNeuralNetworkExperiment):
            self._save_nn_model(experiment, parent_widget)
        else:
            self._save_ml_model(experiment.model, parent_widget)

    def get_experiments_by_task(self, task_type):

        matching_experiments = []

        if hasattr(task_type, 'value'):
            task_value = task_type.value
        else:
            task_value = task_type

        for experiment_id, experiment in self.experiments.items():
            experiment_task = experiment.task

            if (hasattr(experiment_task, 'value') and experiment_task.value == task_value) or \
                    (experiment_task == task_value) or \
                    (str(experiment_task) == str(task_value)):
                matching_experiments.append(experiment)

        self.get_all_task_experiments.emit(matching_experiments)
        return matching_experiments