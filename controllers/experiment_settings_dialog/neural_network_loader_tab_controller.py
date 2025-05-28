import os

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (QFileDialog, QMessageBox)
import tensorflow as tf

from project.controllers.experiment_settings_dialog.tab_controller import TabController
from project.logic.evaluation.task_register import TaskType, NNModelType


class NeuralNetworkLoaderTabController(TabController):
    """
    Controller for the neural network loader widget
    """
    update_model = pyqtSignal()
    update_load_input_data = pyqtSignal()
    update_controller = pyqtSignal(object)

    def __init__(self, experiment, view):
        super().__init__(experiment, view)
        self.model_path = ""
        self.weights_path = ""
        self.loaded_model = None
        self.connect_signals()

    def connect_signals(self):
        self.view.load_button.clicked.connect(self.on_load_clicked)
        self.view.clear_button.clicked.connect(self.on_clear_clicked)
        self.view.browse_weights_button.clicked.connect(self.on_browse_weights_clicked)
        self.view.browse_button.clicked.connect(self.on_browse_clicked)
        self.view.model_load_type_combo.currentIndexChanged.connect(self.on_model_load_type_changed)
        # self.update_model.connect(self.update_model_from_view)

        self.view.model_type_combo.currentIndexChanged.connect(self.view.update_task_list)
        self.view.model_task_combo.currentIndexChanged.connect(self.on_model_task_changed)

    def on_model_task_changed(self):
        self.experiment.task = self.view.model_task_combo.currentText()

    def on_model_type_changed(self):
        self.experiment.task = self.view.model_task_combo.currentText()
        self.update_model.emit()
        self.update_controller.emit(NNModelType(self.view.model_type_combo.currentText()))

    def on_model_load_type_changed(self, index):
        """Handler for model type change"""
        # Show/hide weights fields depending on selected type
        show_weights = (index == 2)  # JSON + Weights
        self.view.weights_path_label.setVisible(show_weights)
        self.view.weights_path_edit.setVisible(show_weights)
        self.view.browse_weights_button.setVisible(show_weights)

        # Update text for model file label
        if index == 0:  # Keras (.h5)
            self.view.file_path_label.setText("Файл моделі (.h5):")
        elif index == 1:  # TensorFlow SavedModel
            self.view.file_path_label.setText("Директорія SavedModel:")
        elif index == 2:  # JSON + Weights
            self.view.file_path_label.setText("Файл структури (.json):")

        # Reset file paths when type changes
        self.view.file_path_edit.clear()
        self.view.weights_path_edit.clear()
        self.model_path = ""
        self.weights_path = ""

    def on_browse_clicked(self):
        """Handler for browse button click to select model file"""
        model_type_index = self.view.model_load_type_combo.currentIndex()

        if model_type_index == 1:  # TensorFlow SavedModel - directory selection
            path = QFileDialog.getExistingDirectory(
                self.view, "Виберіть директорію SavedModel", os.path.expanduser("~")
            )
            if path:
                self.model_path = path
                self.view.file_path_edit.setText(path)
        else:  # Other formats - file selection
            file_filter = ""
            if model_type_index == 0:  # Keras (.h5)
                file_filter = "Keras Models (*.h5);;All Files (*)"
            elif model_type_index == 2:  # JSON
                file_filter = "JSON Files (*.json);;All Files (*)"

            path, _ = QFileDialog.getOpenFileName(
                self.view, "Виберіть файл моделі", os.path.expanduser("~"), file_filter
            )
            if path:
                self.model_path = path
                self.view.file_path_edit.setText(path)

    def on_browse_weights_clicked(self):
        """Handler for browse button click to select weights file"""
        path, _ = QFileDialog.getOpenFileName(
            self.view, "Виберіть файл ваг",
            os.path.expanduser("~"),
            "Weights Files (*.weights *.h5);;All Files (*)"
        )
        if path:
            self.weights_path = path
            self.view.weights_path_edit.setText(path)

    def on_load_clicked(self):
        """Handler for load model button click"""
        if not self.model_path:
            self.view.show_message("Помилка", "Виберіть файл моделі", QMessageBox.Warning)
            return

        model_type_index = self.view.model_load_type_combo.currentIndex()

        # Check for weights presence for JSON + Weights
        if model_type_index == 2 and not self.weights_path:
            self.view.show_message("Помилка", "Виберіть файл ваг", QMessageBox.Warning)
            return

        try:
            # Load model according to selected type
            if model_type_index == 0:  # Keras (.h5)
                self.load_keras_h5_model()
            elif model_type_index == 1:  # TensorFlow SavedModel
                self.load_savedmodel()
            elif model_type_index == 2:  # JSON + Weights
                self.load_json_weights()

            # Display model information
            if self.loaded_model:
                self.display_model_summary()
                self.view.model_loaded.emit(self.loaded_model)
                self.view.show_message("Успіх", "Модель успішно завантажена!")

        except Exception as e:
            error_message = f"Помилка при завантаженні моделі: {str(e)}"
            self.view.show_message("Помилка", error_message, QMessageBox.Critical)

        self.update_model_from_view()

    def load_keras_h5_model(self):
        """Load model from Keras .h5 file format"""
        self.loaded_model = tf.keras.models.load_model(self.model_path)

    def load_savedmodel(self):
        """Load model from SavedModel directory"""
        self.loaded_model = tf.saved_model.load(self.model_path)

    def load_json_weights(self):
        """Load model from JSON structure and weights file"""
        # Load model structure from JSON
        with open(self.model_path, 'r') as json_file:
            model_json = json_file.read()

        # Create model from JSON
        model = tf.keras.models.model_from_json(model_json)

        # Load weights
        model.load_weights(self.weights_path)

        self.loaded_model = model

    def display_model_summary(self):
        """Display information about the loaded model"""
        if not self.loaded_model:
            return

        # Get model information
        model_info = ""

        # Check model type
        if isinstance(self.loaded_model, tf.keras.Model):
            # For Keras models
            stringlist = []
            self.loaded_model.summary(print_fn=lambda x: stringlist.append(x))
            model_info = "\n".join(stringlist)

            # Add additional information
            model_info += "\n\nТип моделі: Keras Model"
            model_info += f"\nКількість шарів: {len(self.loaded_model.layers)}"
            model_info += f"\nВхідна форма: {self.loaded_model.input_shape}"
            model_info += f"\nВихідна форма: {self.loaded_model.output_shape}"

        elif isinstance(self.loaded_model, tf.Module):
            # For SavedModel
            model_info = "Завантажено TensorFlow SavedModel\n\n"
            model_info += f"Шлях: {self.model_path}\n"
            try:
                # Try to get signatures
                signatures = self.loaded_model.signatures
                model_info += f"Сигнатури: {list(signatures.keys())}\n"
            except:
                model_info += "Інформація про сигнатури недоступна\n"

        else:
            # For other types
            model_info = f"Завантажено модель типу: {type(self.loaded_model)}\n"
            model_info += f"Шлях до файлу: {self.model_path}\n"

        self.view.display_model_info(model_info)

    def on_clear_clicked(self):
        """Handler for clear button click"""
        self.view.file_path_edit.clear()
        self.view.weights_path_edit.clear()
        self.view.model_info_text.clear()
        self.model_path = ""
        self.weights_path = ""
        self.loaded_model = None

    def update_model_from_view(self):
        self.experiment.load_type = self.view.model_load_type_combo.currentText()
        self.experiment.model_file_path = self.model_path
        self.experiment.weights_file_path = self.weights_path
        task = TaskType(self.view.model_task_combo.currentText())
        self.experiment.task = task
