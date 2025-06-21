from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QTabWidget, QHBoxLayout, QPushButton, QWidget
from PyQt5.QtCore import pyqtSignal

from project.ui.experiment_settings_dialog.general_tab import GeneralTabWidget
from project.ui.experiment_settings_dialog.hypeparams_tab import HyperparamsTabWidget
from project.ui.experiment_settings_dialog.input_data_tab import InputDataTabWidget
from project.ui.experiment_settings_dialog.metrics_tab import MetricsTabWidget



class ExperimentSettingsWindow(QMainWindow):
    """Main window for experiment settings dialog"""
    window_accepted = pyqtSignal()  # Signal to replace dialog.accept()
    window_rejected = pyqtSignal()  # Signal to replace dialog.reject()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model_tab = None
        self.init_ui()


    def init_ui(self):
        """initialize interface"""
        self.setWindowTitle("Налаштування експерименту")
        self.setMinimumWidth(700)
        self.setMinimumHeight(800)
        self.move(100,100)

        # Create central widget for QMainWindow
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)

        # Створюємо віджет з вкладками
        self.tab_widget = QTabWidget()

        # Створюємо вкладки
        self.general_tab = GeneralTabWidget()
        self.data_tab = InputDataTabWidget()
        self.model_tab = HyperparamsTabWidget()
        self.evaluation_tab = MetricsTabWidget()

        # Додаємо вкладки до віджета
        self.tab_widget.addTab(self.general_tab, "Загальна інформація")
        self.tab_widget.addTab(self.model_tab, "Гіперпараметри")
        self.tab_widget.addTab(self.data_tab, "Вхідні дані")
        self.tab_widget.addTab(self.evaluation_tab, "Оцінка")

        main_layout.addWidget(self.tab_widget)
        self.setCentralWidget(central_widget)

    def accept(self):
        """Equivalent to QDialog's accept method"""
        self.window_accepted.emit()
        self.close()



