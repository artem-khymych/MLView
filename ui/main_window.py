import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
    QFrame, QPushButton, QListWidget, QTextEdit, QWidget, QSplitter, QScrollArea, QGraphicsScene, QLabel
)
from PyQt5.QtCore import Qt, pyqtSignal, QObject

from project.ui.basic_window import BasicWindow
from project.ui.graphics_view import GraphicsView

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MLview")
        self.setGeometry(30, 30, 1000, 1000)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        main_layout = QVBoxLayout(main_widget)

        self.top_panel = QFrame()
        self.top_panel.setFrameShape(QFrame.StyledPanel)
        self.init_top_panel()
        main_layout.addWidget(self.top_panel, 1)

        self.splitter = QSplitter(Qt.Horizontal)
        self.central_area = QFrame()
        self.central_area.setFrameShape(QFrame.StyledPanel)

        self.init_central_area()
        self.init_inspector()
        self.splitter.addWidget(self.inspector_frame)
        self.splitter.addWidget(self.central_area)

        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 3)

        center_layout = QHBoxLayout()
        center_layout.addWidget(self.splitter)

        main_layout.addLayout(center_layout, 10)

        class SignalEmitter(QObject):
            node_created = pyqtSignal(object)
            node_deleted = pyqtSignal(int)
            node_renamed = pyqtSignal(object)
            add_new_experiment = pyqtSignal()
            open_settings = pyqtSignal()

        self.signals = SignalEmitter()

    def init_inspector(self):
        self.inspector_frame = BasicWindow()
        label = QLabel("Інспектор Експериментів")
        label.setStyleSheet("font-weight: bold; font-size: 14px;")
        nodes_list = QListWidget()
        nodes_list.setObjectName("nodes_list")
        self.inspector_frame.layout.addWidget(label)
        self.inspector_frame.layout.addWidget(nodes_list)

    def init_top_panel(self):
        layout = QHBoxLayout(self.top_panel)

        self._setup_menu()
        self.new_experiment_button = QPushButton("Створити експеримент")
        self.new_experiment_button.clicked.connect(self._add_new_experiment)

        layout.addWidget(self.new_experiment_button)
        layout.addStretch()

    def _add_new_experiment(self):
        self.signals.add_new_experiment.emit()

    def init_central_area(self):
        layout = QVBoxLayout(self.central_area)
        self.scene = QGraphicsScene()
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.graphics_view = GraphicsView()
        self.graphics_view.setScene(self.scene)

        self.central_widget = QWidget()
        self.central_layout = QVBoxLayout(self.central_widget)
        self.central_layout.addWidget(self.graphics_view)
        self.scroll_area.setWidget(self.central_widget)

        self.central_area.setLayout(QVBoxLayout())
        self.central_area.layout().addWidget(self.scroll_area)
        self.central_area.setStyleSheet("background-color: white;")

    def _setup_menu(self):
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("Файл")

        self.new_action = file_menu.addAction("Новий проект")
        self.new_action.setShortcut("Ctrl+N")

        self.open_action = file_menu.addAction("Відкрити проект")
        self.open_action.setShortcut("Ctrl+O")

        self.save_action = file_menu.addAction("Зберегти проект")
        self.save_action.setShortcut("Ctrl+S")

        self.save_as_action = file_menu.addAction("Зберегти проект як...")
        self.save_as_action.setShortcut("Ctrl+Shift+S")

        view_menu = menu_bar.addMenu("Вигляд")
        self.fit_action = view_menu.addAction("Підігнати до вмісту")
        self.fit_action.setShortcut("F")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
