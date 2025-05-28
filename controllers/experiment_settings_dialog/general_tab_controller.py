from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QMessageBox, QPushButton
import copy

from project.logic.experiment.experiment import Experiment
from project.logic.experiment.generic_nn_experiment import GenericNeuralNetworkExperiment
from project.logic.experiment.nn_experiment import NeuralNetworkExperiment
from project.ui.experiment_settings_dialog.general_tab import GeneralTabWidget


class GeneralSettingsController(QObject):
    """Controller for general settings tab"""
    experiment_inherited = pyqtSignal(int)

    def __init__(self, experiment: Experiment, view: GeneralTabWidget) -> None:
        super().__init__()
        self.experiment = experiment
        self.view = view
        self.init_view()
        self.connect_signals()

    def init_view(self):
        """Initialize the view's initial state"""
        self.view.experiment_name.setText(self.experiment.name)
        self.view.description.setText(self.experiment.description)
        self.view.method_name.setText(type(self.experiment.model).__name__)

        # Set experiment status
        self.view.update_status(self.experiment.is_finished)

        if hasattr(self.experiment, 'training_time') and self.experiment.is_finished:
            self.view.set_experiment_finished(self.experiment.training_time)
            # Enable save button if experiment is complete
            self.view.save_button.setEnabled(True)

        if isinstance(self.experiment, NeuralNetworkExperiment):
            self.history_button = QPushButton("Переглянути історію")  # "View history"
            self.view.button_layout.addWidget(self.history_button)

            # Show history button immediately if experiment is complete
            if isinstance(self.experiment, GenericNeuralNetworkExperiment) and self.experiment.is_finished:
                self.history_button.setVisible(True)
            else:
                self.history_button.setVisible(False)
        else:
            self.history_button = None

    def update_model_from_view(self):
        """Update model with data from view"""
        self.experiment.name = self.view.experiment_name.text()
        self.experiment.description = self.view.description.toPlainText()
        if self.experiment.is_finished:
            self.view.start_button.setText("Перезапустити")  # "Restart"

    def set_experiment_description(self):
        self.experiment.description = self.view.description.toPlainText()

    def set_experiment_name(self, name: str):
        self.view.experiment_name.setText(name)

    def connect_signals(self):
        """Connect signal handlers"""
        self.view.evaluate_clicked.connect(self.experiment.evaluate)
        self.experiment.experiment_finished.connect(self.on_experiment_finished)
        self.view.description.textChanged.connect(self.set_experiment_description)
        # Connect inheritance button to appropriate method
        self.view.inherit_button.clicked.connect(self.on_experiment_inherited)

    def on_experiment_finished(self, training_time):
        """Experiment completion handler"""
        print(f"Experiment completed in {training_time} seconds")
        self.experiment.is_finished = True
        self.view.time_label.setText(f"На тренування витрачено {str(training_time)} секунд")  # "Training took {time} seconds"
        QMessageBox.information(self.view, "Успіх",  # "Success"
                                "Модель успішно натренована.")  # "Model trained successfully"
        self.view.set_experiment_finished(training_time)
        if isinstance(self.experiment, GenericNeuralNetworkExperiment):
            self.history_button.setVisible(True)
        self.view.save_button.setEnabled(True)

    def on_experiment_inherited(self):
        """Experiment inheritance button click handler"""
        # Send signal with current experiment ID to create new inherited experiment
        self.experiment_inherited.emit(self.experiment.id)
        QMessageBox.information(self.view, "Успадкування",  # "Inheritance"
            f"Створено новий експеримент на основі '{self.experiment.name}'")
        # f"Created new experiment based on '{self.experiment.name}'"