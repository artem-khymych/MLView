from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QVBoxLayout, QGridLayout, QGroupBox, QLabel, QComboBox, QPushButton, QLineEdit, QTextEdit, \
    QHBoxLayout, QMessageBox, QWidget

from project.logic.evaluation.task_register import NNModelType, ModelTaskRegistry


class NeuralNetworkLoaderTabWidget(QWidget):
    """
    Widget for loading neural networks from various file formats
    """
    model_loaded = pyqtSignal(object)  # Signal that transmits the loaded model

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Завантажувач нейронних мереж")
        self.setMinimumWidth(600)

        # Main layout
        main_layout = QVBoxLayout()

        # File loading group
        file_group = QGroupBox("Завантаження файлів моделі")
        file_layout = QGridLayout()

        # Model type selection
        self.model_type_label = QLabel("Тип моделі:")
        self.model_load_type_combo = QComboBox()
        self.model_load_type_combo.addItems(["Keras (.h5)", "TensorFlow SavedModel",
                                        "JSON + Weights"])

        # File selection fields
        self.file_path_label = QLabel("Файл моделі:")
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setReadOnly(True)
        self.browse_button = QPushButton("Огляд...")


        # Additional fields for JSON + Weights format
        self.weights_path_label = QLabel("Файл ваг:")
        self.weights_path_edit = QLineEdit()
        self.weights_path_edit.setReadOnly(True)
        self.browse_weights_button = QPushButton("Огляд...")

        # Initially hide weights fields
        self.weights_path_label.setVisible(False)
        self.weights_path_edit.setVisible(False)
        self.browse_weights_button.setVisible(False)

        self.model_type_combo = QComboBox()
        modeltypes = NNModelType
        self.model_type_combo.addItems(
            [model_type.value for model_type in modeltypes]
        )

        self.model_task_combo = QComboBox()
        tasks = ModelTaskRegistry().get_tasks_for_model((self.model_type_combo.currentText()))

        self.model_task_combo.addItems(
            [eval_type.task_type.value for eval_type in tasks]
        )

        # Arrange elements in the file loading group
        file_layout.addWidget(self.model_type_label, 0, 0)
        file_layout.addWidget(self.model_load_type_combo, 0, 1, 1, 2)
        file_layout.addWidget(self.file_path_label, 1, 0)
        file_layout.addWidget(self.file_path_edit, 1, 1)
        file_layout.addWidget(self.browse_button, 1, 2)
        file_layout.addWidget(self.weights_path_label, 2, 0)
        file_layout.addWidget(self.weights_path_edit, 2, 1)
        file_layout.addWidget(self.browse_weights_button, 2, 2)
        file_layout.addWidget(QLabel("Тип моделі:"), 4, 1)
        file_layout.addWidget(self.model_type_combo, 4, 2)
        file_layout.addWidget(QLabel("Задача:"), 5, 1)
        file_layout.addWidget(self.model_task_combo, 5, 2)

        file_group.setLayout(file_layout)

        # Model information group
        info_group = QGroupBox("Інформація про модель")
        info_layout = QVBoxLayout()

        self.model_info_text = QTextEdit()
        self.model_info_text.setReadOnly(True)
        self.model_info_text.setPlaceholderText("Інформація про модель буде відображена тут після завантаження")

        info_layout.addWidget(self.model_info_text)
        info_group.setLayout(info_layout)

        # Control buttons
        buttons_layout = QHBoxLayout()
        self.load_button = QPushButton("Завантажити модель")

        self.clear_button = QPushButton("Очистити")

        buttons_layout.addWidget(self.load_button)
        buttons_layout.addWidget(self.clear_button)

        # Add all groups and layouts to the main layout
        main_layout.addWidget(file_group)
        main_layout.addWidget(info_group)
        main_layout.addLayout(buttons_layout)

        self.setLayout(main_layout)

    def show_message(self, title, message, icon=QMessageBox.Information):
        """Display an informational message"""
        msg_box = QMessageBox(self)
        msg_box.setIcon(icon)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.exec_()

    def display_model_info(self, info_text):
        """Display model information in the text field"""
        self.model_info_text.setPlainText(info_text)

    def update_task_list(self):
        self.model_task_combo.clear()
        tasks = ModelTaskRegistry().get_tasks_for_model((self.model_type_combo.currentText()))

        self.model_task_combo.addItems(
            [eval_type.task_type.value for eval_type in tasks]
        )

