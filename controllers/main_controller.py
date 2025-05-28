from PyQt5.QtWidgets import QMainWindow, QDialog

from project.controllers.experiment_settings_dialog.experiment_settings_controller import ExperimentSettingsController
from project.controllers.inspector_controller import InspectorController
from project.controllers.task_selector_controller import TaskSelectorController
from project.controllers.workspace_manager import WorkspaceManager
from project.logic.experiment_manager import ExperimentManager
from project.logic.modules.models_manager import ModelsManager
from project.ui.experiment_settings_dialog.experiment_comparison_dialog import ExperimentComparisonDialog
from project.ui.experiment_settings_dialog.experiment_settings_dialog import ExperimentSettingsWindow

from project.ui.main_window import MainWindow


class MainController:
    def __init__(self):
        self.workspace_manager = WorkspaceManager()
        self.view = MainWindow()
        self.models_manager = ModelsManager()
        self.inspector_controller = InspectorController(self.view.inspector_frame,
                                                        self.view.scene,
                                                        self.view.graphics_view)
        self.task_selector_controller = TaskSelectorController(self.view)
        self.experiment_manager = ExperimentManager()

        self.workspace_manager.set_experiment_manager(self.experiment_manager)
        self.workspace_manager.set_node_controller(self.inspector_controller.node_controller)
        self.workspace_manager.set_work_area(self.view.graphics_view)  # Passing the work area

        self.connect_signals()

    def connect_signals(self):
        self.task_selector_controller.request_models_dict.connect(self.models_manager.create_models_dict)
        self.models_manager.models_dict_ready.connect(self.task_selector_controller.handle_models_dict_response)
        self.view.signals.add_new_experiment.connect(self.inspector_controller.node_controller.create_node)
        self.inspector_controller.node_controller.node_created.connect(self.experiment_manager.get_node)
        self.view.signals.add_new_experiment.connect(self.task_selector_controller.show_approach_selection)
        self.task_selector_controller.send_ml_model.connect(self.experiment_manager.get_ml_model)
        self.inspector_controller.node_controller.nodeInfoOpened.connect(self._show_experiment_settings_dialog)

        # Connecting the experiment inheritance signal from the node controller
        self.inspector_controller.node_controller.experiment_inherited.connect(self._handle_experiment_inheritance)

        self.task_selector_controller.send_nn_model.connect(self.experiment_manager.create_nn_experiment)

        self.view.fit_action.triggered.connect(self.workspace_manager.fit_view_to_content)
        self.view.save_as_action.triggered.connect(lambda: self.workspace_manager.save_project_as(self.view))
        self.view.save_action.triggered.connect(lambda: self.workspace_manager.save_project(self.view))
        self.view.open_action.triggered.connect(lambda: self.workspace_manager.open_project(self.view))
        self.view.new_action.triggered.connect(self.workspace_manager.new_project)

        self.inspector_controller.node_controller.update_experiment_name.connect(self.experiment_manager.update_name)


    def _show_experiment_settings_dialog(self, node_id):
        """Function to display the experiment settings dialog"""
        dialog = ExperimentSettingsWindow(self.view)
        experiment = self.experiment_manager.get_experiment(node_id)
        self.experiment_settings_controller = ExperimentSettingsController(experiment, dialog)

        # Connecting the inheritance signal from settings dialog
        self.experiment_settings_controller.experiment_inherited.connect(self._handle_experiment_inheritance)
        self.experiment_settings_controller.metrics_controller.compare_experiments.connect(
            self.experiment_manager.show_comparison_dialog)

        #self.experiment_settings_controller.request_experiment_update.connect(
        #    self.experiment_manager.update_nn_experiment)  # change nn experiment class in order to task

        self.experiment_settings_controller.show()
        self.experiment_settings_controller.window.general_tab.save_button.clicked.connect(
            lambda: self.experiment_manager.save_model(experiment, self.view))

        self.experiment_settings_controller.metrics_controller.compare_all.connect(self.experiment_manager.get_experiments_by_task)
        self.experiment_manager.get_all_task_experiments.connect(ExperimentComparisonDialog.create_dialog_with_filtered_experiments)

    def _handle_experiment_inheritance(self, parent_id):
        """Handler for experiment inheritance signal"""
        # Create new node and inherit experiment data
        parent_experiment = self.experiment_manager.get_experiment(parent_id)

        # Create new node through node controller
        new_node = self.inspector_controller.node_controller.create_inherited_node(parent_id)

        # After node creation, inherit the experiment
        if new_node and parent_experiment:
            # Experiment manager will create new experiment with inherited data
            self.experiment_manager.inherit_experiment_from(parent_id, new_node.id)

    def show(self):
        self.view.show()
