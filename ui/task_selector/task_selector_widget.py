from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton


class TaskSelectorWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Main layout
        main_layout = QVBoxLayout(self)

        # Title
        title_label = QLabel("Вибір типу експерименту")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont('Arial', 16))
        main_layout.addWidget(title_label)

        # Select Button
        self.select_btn = QPushButton("Вибрати")
        self.select_btn.setMinimumHeight(50)
        self.select_btn.setFont(QFont('Arial', 14))
        main_layout.addWidget(self.select_btn)