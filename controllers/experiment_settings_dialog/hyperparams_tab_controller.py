from abc import ABC

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QMessageBox

from project.controllers.experiment_settings_dialog.ml_tuning_dialog import show_param_tuning_dialog
from project.controllers.experiment_settings_dialog.nn_tuning_dialog import show_nn_tuning_dialog
from project.controllers.experiment_settings_dialog.tab_controller import TabController
from project.logic.experiment.experiment import Experiment
from project.logic.experiment.generic_nn_experiment import GenericNeuralNetworkExperiment
from project.ui.experiment_settings_dialog.hypeparams_tab import HyperparamsTabWidget


class HyperparamsTabController(TabController):
    """Controller for model parameters tab"""
    get_input_data_for_tuning = pyqtSignal()

    def __init__(self, experiment: Experiment, view: HyperparamsTabWidget):
        super().__init__(experiment, view)
        self.init_view()
        self.connect_signals()

    def connect_signals(self):
        self.view.tune_params.clicked.connect(self._tune_params_start)

    def init_view(self):
        """Initialize the view with current parameters"""
        self.view.params_widget.populate_table(self.experiment.params)

    def _tune_params_start(self):
        """Start parameters tuning process"""
        self.get_input_data_for_tuning.emit()

        try:
            X_train, y_train = self.experiment.get_params_for_tune()
            if not isinstance(X_train, type(None)):
                if isinstance(self.experiment, GenericNeuralNetworkExperiment):
                    self.experiment.load_model_from_file()
                    model = self.experiment.model
                    compile_params, fit_params = show_nn_tuning_dialog(model, X_train, y_train)
                    self.experiment.params["model_params"] = compile_params
                    self.experiment.params["fit_params"] = fit_params
                    self.view.params_widget.update_parameters(self.experiment.params)
                else:
                    model = self.experiment.model
                    params = self.experiment.params
                    best_params = show_param_tuning_dialog(model, params, X_train, y_train)
                    self.experiment.params = best_params
                    self.view.params_widget.update_parameters(best_params)
            else:
                QMessageBox.critical(self.view, "Помилка",  # "Error"
                                     "Налаштуйте вхідний датасет")  # "Configure input dataset"
                return None
        except Exception as e:
            QMessageBox.critical(self.view, "Помилка",  # "Error"
                                 str(e))

    def _update_params(self):
        """Update parameters from view to model"""
        self.experiment.params = self.view.params_widget.get_current_parameters()
        print(self.view.params_widget.get_current_parameters())

    def update_model_from_view(self):
        """Update model with parameters from view"""
        self._update_params()