import sys

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
    QFrame, QPushButton, QListWidget, QTextEdit, QWidget, QSplitter, QGraphicsProxyWidget, QSpacerItem, QSizePolicy,
    QTableWidget
)
from PyQt5.QtCore import Qt, pyqtSignal
from qtpy import QtGui


class BasicWindow(QFrame):
    """Class for representing a basic layout with minimize button for central area widgets"""
    changeSizeRequested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.MINIMUM_SIZE = 20
        self.setMinimumSize(self.MINIMUM_SIZE, self.height())
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)



