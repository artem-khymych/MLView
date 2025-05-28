from PyQt5.QtCore import pyqtSignal, QObject
from PyQt5.QtWidgets import QMessageBox

from project.controllers.experiment_settings_dialog.general_tab_controller import GeneralSettingsController
from project.controllers.experiment_settings_dialog.hyperparams_tab_controller import HyperparamsTabController
from project.controllers.experiment_settings_dialog.input_data_tab_controller import InputDataTabController
from project.controllers.experiment_settings_dialog.metrics_tab_controller import MetricsTabController
from project.controllers.experiment_settings_dialog.neural_network_loader_tab_controller import \
    NeuralNetworkLoaderTabController
from project.logic.experiment.experiment import Experiment
from project.logic.experiment.generic_nn_experiment import GenericNeuralNetworkExperiment
from project.logic.experiment.nn_experiment import NeuralNetworkExperiment

from project.ui.experiment_settings_dialog.experiment_settings_dialog import ExperimentSettingsWindow

from project.ui.experiment_settings_dialog.neural_network_load_tab import NeuralNetworkLoaderTabWidget
from project.ui.experiment_settings_dialog.training_history_dialog import TrainingHistoryDialog


class ExperimentSettingsController(QObject):
    """Main controller for experiment settings window"""
    experiment_inherited = pyqtSignal(int)
    window_closed = pyqtSignal(bool, object)
    request_experiment_update = pyqtSignal(object, object)

    def __init__(self, experiment: Experiment, window: ExperimentSettingsWindow):
        super().__init__()
        self.experiment = experiment
        self.window = window
        self.result_accepted = False
        self.input_params = None

        # Creating controllers for tabs
        self.general_controller = GeneralSettingsController(experiment, window.general_tab)
        self.input_data_controller = InputDataTabController(self.experiment, self.window.data_tab)
        self.metrics_controller = MetricsTabController(experiment, window.evaluation_tab)
        self.hyperparams_controller = HyperparamsTabController(self.experiment, self.window.model_tab)

        # self.prepare_nn_loader()
        self.connect_signals()

    def prepare_nn_loader(self):
        if isinstance(self.experiment, GenericNeuralNetworkExperiment):
            self.window.nn_loader_tab = NeuralNetworkLoaderTabWidget()
            self.window.tab_widget.addTab(self.window.nn_loader_tab, "Завантажити модель")
            self.nn_loader_controller = NeuralNetworkLoaderTabController(self.experiment, self.window.nn_loader_tab)
            if self.experiment.parent:
                self.nn_loader_controller = None
                self.window.tab_widget.removeTab(4)

        else:
            self.nn_loader_controller = None
        return

    def connect_signals(self):
        self.window.window_accepted.connect(self.on_accept)
        self.window.window_rejected.connect(self.on_cancel)
        self.general_controller.view.experiment_started.connect(self.check_settings_and_run_experiment)

        # evaluation started, update metrics and go to metrics tab
        self.experiment.experiment_evaluated.connect(self.metrics_controller.on_metrics_updated)
        self.experiment.experiment_evaluated.connect(lambda: self.window.tab_widget.setCurrentIndex(3))

        # Connect inheritance button to handler method
        self.general_controller.experiment_inherited.connect(self.on_experiment_inherited)
        if self.general_controller.history_button:
            self.general_controller.history_button.clicked.connect(self.on_show_history)
        self.hyperparams_controller.get_input_data_for_tuning.connect(self.update_model_from_all_views)

    def on_update_controller(self, model_type):
        self.hyperparams_controller = HyperparamsTabController(self.experiment, self.window.model_tab)
        self.general_controller.view.method_name.setText(model_type.value)

    def on_show_history(self):
        dialog = TrainingHistoryDialog()
        dialog.show_history(self.experiment.history)

    def on_experiment_inherited(self, parent_id):
        """Forward inheritance signal"""
        # Create signal to pass up the hierarchy
        self.experiment_inherited.emit(parent_id)

    def check_settings_and_run_experiment(self):
        self.update_model_from_all_views()
        if self.check_settings():
            self.experiment.run()
        else:
            return

    def on_cancel(self):
        self.result_accepted = False
        self.window_closed.emit(False, None)

    def on_accept(self):
        self.update_model_from_all_views()
        self.result_accepted = True
        self.input_params = self.input_data_controller.get_input_params()
        self.window_closed.emit(True, self.input_params)

    def check_settings(self):
        if isinstance(self.experiment, NeuralNetworkExperiment):
            return self.check_nn_experiment_settings()
        elif isinstance(self.experiment, Experiment):
            return self.check_experiment_settings()

    def _check_input_files(self):
        """Validate all input data"""
        if (self.input_data_controller.input_data_params.mode == 'single_file'
                and not self.input_data_controller.input_data_params.single_file_path):
            QMessageBox.warning(self.window, "Помилка", "Будь ласка, виберіть файл даних на вкладці 'Дані'.")
            self.window.tab_widget.setCurrentIndex(2)
            return False
        elif self.input_data_controller.input_data_params.mode == 'multi_files' and (
                not self.input_data_controller.input_data_params.x_train_file_path
                or not self.input_data_controller.input_data_params.y_train_file_path
                or not self.input_data_controller.input_data_params.x_test_file_path
                or not self.input_data_controller.input_data_params.y_test_file_path
        ):
            QMessageBox.warning(self.window, "Помилка",
                                "Будь ласка, виберіть обидва файли для тренування і тестування на вкладці 'Дані'.")
            self.window.tab_widget.setCurrentIndex(2)  # Switch to "Data" tab
            return False

        # Check data from "General Settings" tab
        if not self.window.general_tab.experiment_name.text().strip():
            QMessageBox.warning(self.window, "Помилка", "Будь ласка, введіть назву експерименту.")
            self.window.tab_widget.setCurrentIndex(0)  # Switch to "General Settings" tab
            return False

        return True

    def check_experiment_settings(self):
        if not self._check_input_files():
            return False

        try:
            if isinstance(self.experiment, Experiment):
                model = type(self.experiment.model)().set_params(**self.experiment.params)
        except Exception as e:
            QMessageBox.warning(self.window, "Невірні параметри", f"Виникла помилка у налаштованих параметрах:\n {e}")
            return False
        if self.validate_params_strict(model, self.experiment.params):
            return True
        else:
            return False

    def check_nn_experiment_settings(self):
        if self._check_input_files():
            return True

    def validate_params_strict(self, model_class, params):
        from sklearn.utils._param_validation import validate_parameter_constraints

        if not hasattr(model_class, "_parameter_constraints"):
            return True

        constraints = model_class._parameter_constraints
        try:
            validate_parameter_constraints(constraints, params, caller_name=model_class.__class__.__name__)
        except Exception as e:
            QMessageBox.warning(self.window, "Невірні параметри", f"Виникла помилка у налаштованих параметрах:\n {e}")
            return False

        return True

    def update_model_from_all_views(self):
        """Update model with data from all views"""

        self.general_controller.update_model_from_view()
        self.hyperparams_controller.update_model_from_view()
        self.input_data_controller.update_model_from_view()

    def show(self):
        """Show the window and wait for the result"""
        self.window.show()