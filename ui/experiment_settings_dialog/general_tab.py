from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QTextEdit, QPushButton


class GeneralTabWidget(QWidget):
    """Widget for general settings tab"""

    # Signal for starting experiment
    experiment_started = pyqtSignal()
    evaluate_clicked = pyqtSignal()


    def __init__(self, parent=None):
        super().__init__(parent)
        self.description = ""
        self.experiment_name = ""
        self.is_finished = False
        self.training_time = None
        self.init_ui()

    def init_ui(self):
        """Initialize user interface"""
        layout = QVBoxLayout()

        # Experiment name
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Назва експерименту:"))
        self.experiment_name = QLineEdit("Новий експеримент")
        name_layout.addWidget(self.experiment_name)
        layout.addLayout(name_layout)

        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Method/Model:"))
        self.method_name = QLabel("")
        model_layout.addWidget(self.method_name)
        layout.addLayout(model_layout)

        # Experiment description
        layout.addWidget(QLabel("Опис експерименту:"))
        self.description = QTextEdit()

        layout.addWidget(self.description)

        # Experiment status
        self.status_layout = QHBoxLayout()
        self.status_label = QLabel("Статус експерименту:")
        self.status_value = QLabel("Не запущено")
        self.status_layout.addWidget(self.status_label)
        self.status_layout.addWidget(self.status_value)
        layout.addLayout(self.status_layout)

        # Training time display
        self.time_layout = QHBoxLayout()
        self.time_label = QLabel("Час навчання:")
        self.time_value = QLabel("-")
        self.time_layout.addWidget(self.time_label)
        self.time_layout.addWidget(self.time_value)
        layout.addLayout(self.time_layout)

        # Button for starting experiment
        self.button_layout = QHBoxLayout()
        self.start_button = QPushButton("Розпочати")

        self.evaluate_button = QPushButton("Оцінити")
        self.training_time = QLabel("")
        self.evaluate_button.setEnabled(False)

        self.inherit_button = QPushButton("Успадкувати")
        self.training_time = QLabel("")
        self.inherit_button.setEnabled(False)

        self.save_button = QPushButton("Зберегти модель")
        self.save_button.setEnabled(False)

        # Add green triangle as icon
        self.start_button.setIcon(QIcon.fromTheme("media-playback-start"))  # Standard play icon

        self.start_button.clicked.connect(self.on_start_clicked)
        self.evaluate_button.clicked.connect(self.on_evaluate_clicked)

        self.button_layout.addWidget(self.start_button)
        self.button_layout.addWidget(self.evaluate_button)
        self.button_layout.addWidget(self.training_time)
        self.button_layout.addWidget(self.inherit_button)
        self.button_layout.addWidget(self.save_button)

        self.button_layout.addStretch()
        layout.addLayout(self.button_layout)

        layout.addStretch()

        self.setLayout(layout)

    def on_start_clicked(self):
        """Handler for start button click"""

        self.experiment_started.emit()

    def on_evaluate_clicked(self):
        self.evaluate_clicked.emit()


    def set_experiment_finished(self, training_time):
        """Set experiment completion status"""
        self.is_finished = True
        self.training_time = training_time
        self.status_value.setText("Завершено")
        self.evaluate_button.setEnabled(True)
        self.inherit_button.setEnabled(True)

    def update_status(self, is_finished, training_time=None):
        """Update experiment status"""
        self.is_finished = is_finished
        if is_finished:
            self.status_value.setText("Завершено")
            if training_time is not None:
                self.training_time = training_time
                self.time_value.setText(f"{training_time} с")
        else:
            self.status_value.setText("Не запущено")
            self.start_button.setEnabled(True)

    def apply_style(self):
        """Apply styles for scientific-practical application"""
        # General widget style
        self.setStyleSheet("""
            QWidget {
                background-color: #F5F7FA;
                color: #2C3E50;
                font-family: 'Segoe UI', 'Arial', sans-serif;
            }
            QLabel {
                font-size: 11pt;
                font-weight: 500;
                padding: 2px;
            }
            QLineEdit, QTextEdit {
                border: 1px solid #BDC3C7;
                border-radius: 3px;
                padding: 5px;
                background-color: white;
                selection-background-color: #3498DB;
            }
            QTextEdit {
                font-size: 10pt;
            }
            QPushButton {
                background-color: #3498DB;
                color: white;
                border-radius: 3px;
                padding: 6px 12px;
                font-weight: bold;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #2980B9;
            }
            QPushButton:pressed {
                background-color: #1F618D;
            }
            QPushButton:disabled {
                background-color: #BDC3C7;
                color: #7F8C8D;
            }
        """)

        # Specific styles for statuses
        if self.is_finished:
            self.status_value.setStyleSheet("color: #27AE60; font-weight: bold;")
        else:
            self.status_value.setStyleSheet("color: #E74C3C; font-weight: bold;")

        # Styles for buttons with different functions
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #2ECC71;
                color: white;
                border-radius: 3px;
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #27AE60;
            }
            QPushButton:pressed {
                background-color: #1E8449;
            }
            QPushButton:disabled {
                background-color: #BDC3C7;
                color: #7F8C8D;
            }
        """)

        self.evaluate_button.setStyleSheet("""
            QPushButton {
                background-color: #3498DB;
                color: white;
                border-radius: 3px;
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980B9;
            }
            QPushButton:pressed {
                background-color: #1F618D;
            }
            QPushButton:disabled {
                background-color: #BDC3C7;
                color: #7F8C8D;
            }
        """)

        self.inherit_button.setStyleSheet("""
            QPushButton {
                background-color: #9B59B6;
                color: white;
                border-radius: 3px;
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #8E44AD;
            }
            QPushButton:pressed {
                background-color: #6C3483;
            }
            QPushButton:disabled {
                background-color: #BDC3C7;
                color: #7F8C8D;
            }
        """)

        self.save_button.setStyleSheet("""
            QPushButton {
                background-color: #F39C12;
                color: white;
                border-radius: 3px;
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #E67E22;
            }
            QPushButton:pressed {
                background-color: #D35400;
            }
            QPushButton:disabled {
                background-color: #BDC3C7;
                color: #7F8C8D;
            }
        """)