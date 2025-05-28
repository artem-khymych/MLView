from typing import Dict

from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QPushButton, QLabel, QVBoxLayout, QDialog, QRadioButton, QButtonGroup, QListWidget

from project.controllers.experiment_settings_dialog.neural_network_loader_tab_controller import \
    NeuralNetworkLoaderTabController
from project.logic.modules import task_names
from project.ui.experiment_settings_dialog.neural_network_load_tab import NeuralNetworkLoaderTabWidget
from project.ui.task_selector.dynamic_button_dialog import DynamicButtonDialog
from project.ui.parameter_editor_widget import ParameterEditorWidget
from project.controllers.parameter_editor_controller import ParameterEditorController

from project.logic.evaluation.task_register import TaskType, NNModelType


class TaskSelectorController(QObject):
    request_models_dict = pyqtSignal(str)

    own_nn_selected = pyqtSignal(str)
    send_ml_model = pyqtSignal(str, object, object)
    send_nn_model = pyqtSignal(object, str, str, str)

    _instance = None  # Змінна класу для зберігання екземпляра

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, sender):
        super().__init__()
        self.sender = sender
        self.selected_task: str = ""
        self.nn_loader_dialog = None
        self.nn_loader_controller = None

    def show_approach_selection(self):
        approach_dict = {
            "Класичне МН": 1,
            "Нейронна мережа": 2
        }
        approach_dialog = DynamicButtonDialog("Вибір типу експерименту", approach_dict, self.sender)
        result = approach_dialog.exec_()

        if result == 1:  # Classical ML
            self.show_learning_type_selection()
        elif result == 2:  # Neural Networks
            self.show_neural_network_selection()

    def show_learning_type_selection(self):
        learning_type_dict = {
            "Навчання із вчителем": 1,
            "Навчання без вчителя": 2
        }
        learning_type_dialog = DynamicButtonDialog("Вибір підходу до навчання", learning_type_dict, self.sender)
        result = learning_type_dialog.exec_()

        if result == 1:  # Supervised Learning
            self.show_supervised_task_selection()
        elif result == 2:  # Unsupervised Learning
            self.show_unsupervised_task_selection()

    def show_supervised_task_selection(self):
        tasks = {
            task_names.CLASSIFICATION: 1,
            task_names.REGRESSION: 2
        }
        self.show_task_selection_dialog("Вибір задачі із вчителем", tasks)

    def show_unsupervised_task_selection(self):
        tasks = {
            task_names.CLUSTERING: 1,
            task_names.DIMENSIONALITY_REDUCTION: 2,
            task_names.ANOMALY_DETECTION: 3,
            task_names.DENSITY_ESTIMATION: 4
        }
        self.show_task_selection_dialog("Вибір задачі без вчителя", tasks)

    def show_neural_network_selection(self):
        nn_types = {
            "Scikit-learn MLP models": 1,
            "Import own": 2
        }
        self.show_task_selection_dialog("Вибір типу нейромережі", nn_types)

    def show_task_selection_dialog(self, title, tasks):
        dialog = QDialog(self.sender)
        dialog.setWindowTitle(title)
        dialog.setGeometry(200, 200, 400, 500)

        layout = QVBoxLayout(dialog)

        # Button Group for Tasks
        task_group = QButtonGroup(dialog)

        for task_text, task_value in tasks.items():
            radio_btn = QRadioButton(task_text)
            radio_btn.setMinimumHeight(100)
            radio_btn.setFont(QFont('Arial', 12))
            task_group.addButton(radio_btn, task_value)
            layout.addWidget(radio_btn)

        # Confirm Button
        confirm_btn = QPushButton("Підтвердити")
        confirm_btn.clicked.connect(lambda: self.handle_task_selection(task_group, dialog))
        layout.addWidget(confirm_btn)

        dialog.exec_()

    def handle_task_selection(self, group, dialog):
        selected_button = group.checkedButton()
        selected_task = group.checkedButton().text()
        self.selected_task = selected_task

        if selected_task == "Import own":
            dialog.accept()
            self.show_nn_loader_dialog()
            return

        if selected_button:
            placeholder_dialog = QDialog(self.sender)
            placeholder_dialog.setWindowTitle("Підтвердження")
            placeholder_dialog.setGeometry(200, 200, 400, 300)

            layout = QVBoxLayout(placeholder_dialog)

            # Title with selected option
            title_label = QLabel(f"Обрано: {selected_task}")
            title_label.setAlignment(Qt.AlignCenter)
            title_label.setFont(QFont('Arial', 14))
            layout.addWidget(title_label)
            dialog.accept()

            self.request_models_dict.emit(selected_task)

    def show_nn_loader_dialog(self):
        """Show dialog for loading neural network model"""
        loader_dialog = QDialog(self.sender)
        loader_dialog.setWindowTitle("Завантаження нейромережі")
        loader_dialog.setGeometry(200, 200, 1000, 800)

        layout = QVBoxLayout(loader_dialog)

        # Create the loader widget and controller
        nn_loader_widget = NeuralNetworkLoaderTabWidget()

        # Create a dummy experiment object
        class DummyExperiment:
            def __init__(self):
                self.task = None
                self.load_type = None
                self.model_file_path = None
                self.weights_file_path = None

        dummy_experiment = DummyExperiment()

        # Create the controller
        self.nn_loader_controller = NeuralNetworkLoaderTabController(dummy_experiment, nn_loader_widget)

        # Connect the model loaded signal
        # Add close button
        buttons_layout = QVBoxLayout()
        save_btn = QPushButton("Перейти до Налаштування")
        save_btn.clicked.connect(self.on_nn_model_loaded)
        buttons_layout.addWidget(save_btn)

        # Add widget to layout
        layout.addWidget(nn_loader_widget)
        layout.addLayout(buttons_layout)

        self.nn_loader_dialog = loader_dialog
        loader_dialog.exec_()

    def on_nn_model_loaded(self, model):
        """Handle when a neural network model is loaded"""
        if self.nn_loader_controller and self.nn_loader_dialog:
            # Get information from the controller
            task = self.nn_loader_controller.experiment.task
            model_file_path = self.nn_loader_controller.experiment.model_file_path
            weights_file_path = self.nn_loader_controller.experiment.weights_file_path
            load_type = self.nn_loader_controller.experiment.load_type
            # Send data to experiment manager
            #task_name = task.value if isinstance(task, TaskType) else str(task)
            self.send_nn_model.emit(task, model_file_path, weights_file_path, load_type)

            # Close the dialog
            self.nn_loader_dialog.accept()

    def show_model_selection_dialog(self, models_dict: Dict):
        """
            Show a dialog with all models from the dictionary and return the selected model

            Args:
                models_dict: Dictionary with model names as keys and model classes as values

            Returns:
                Instance of the selected model or None if canceled
            """

        dialog = QDialog(self.sender)
        dialog.setWindowTitle("Select Model")
        dialog.setGeometry(200, 200, 500, 600)

        # Create layout
        layout = QVBoxLayout(dialog)

        # Add title
        title = QLabel("Виберіть модель")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont('Arial', 14, QFont.Bold))
        layout.addWidget(title)

        # Create list widget for models
        models_list = QListWidget()
        models_list.setFont(QFont('Arial', 12))

        # Sort model names for better usability
        model_names = sorted(models_dict.keys())

        # Add models to list
        for model_name in model_names:
            models_list.addItem(model_name)

        # Set minimum height for better visibility
        models_list.setMinimumHeight(400)
        layout.addWidget(models_list)

        # Add buttons
        buttons_layout = QVBoxLayout()

        select_btn = QPushButton("Обрати")
        select_btn.setFont(QFont('Arial', 12))
        select_btn.setMinimumHeight(40)

        buttons_layout.addWidget(select_btn)
        layout.addLayout(buttons_layout)

        # Initialize result
        selected_model = None

        def on_select():
            nonlocal selected_model
            current_item = models_list.currentItem()
            if current_item:
                model_name = current_item.text()
                model_class = models_dict[model_name]
                selected_model = model_class()
                dialog.accept()

        select_btn.clicked.connect(on_select)

        # Double click to select
        models_list.itemDoubleClicked.connect(lambda item: on_select())

        # Show dialog
        result = dialog.exec_()

        # Return selected model or None if canceled
        return selected_model if result == QDialog.Accepted else None

    def handle_models_dict_response(self, models_dict):
        choosen_model = (self.show_model_selection_dialog(models_dict))
        self.send_data_to_experiment_manager(choosen_model, choosen_model.get_params())

    def send_data_to_experiment_manager(self, model, params):
        print(model, params)
        self.send_ml_model.emit(self.selected_task, model, params)