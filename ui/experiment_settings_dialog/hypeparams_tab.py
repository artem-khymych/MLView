from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSizePolicy, QPushButton

from project.ui.parameter_editor_widget import ParameterEditorWidget


class HyperparamsTabWidget(QWidget):
    """Tab widget for hyperparameters."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.params_widget = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.params_widget = ParameterEditorWidget()
        self.params_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.params_widget)

        self.tune_params = QPushButton("Підібрати параметри")
        layout.addWidget(self.tune_params)

        self.setLayout(layout)
