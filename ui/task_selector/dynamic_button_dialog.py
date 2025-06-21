from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QDialog, QPushButton, QHBoxLayout


class DynamicButtonDialog(QDialog):
    def __init__(self, title, button_dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setGeometry(200, 200, 600, 400)

        main_layout = QHBoxLayout(self)
        self.setWindowFlags(Qt.Dialog | Qt.CustomizeWindowHint | Qt.WindowTitleHint)
        for btn_text, btn_value in button_dict.items():
            button = QPushButton(btn_text)
            button.setMinimumSize(250, 300)
            button.setFont(QFont('Arial', 16))
            button.clicked.connect(lambda _, val=btn_value: self.done(val))
            main_layout.addWidget(button)

