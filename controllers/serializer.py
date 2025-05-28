import os
import pickle
from typing import Dict
from PyQt5.QtCore import QObject, pyqtSignal, QPointF, pyqtSlot
from project.logic.experiment.experiment import Experiment
from project.logic.experiment.generic_nn_experiment import GenericNeuralNetworkExperiment
from project.logic.experiment_manager import ExperimentManager

class WorkspaceSerializer(QObject):
    """
    Class for serializing and deserializing the experiment workspace.
    Stores all information about experiments, nodes and connections between them.
    """

    # Signals for operation status notifications
    workspace_saved = pyqtSignal(str)  # Signal emitted when workspace is saved (path)
    workspace_loaded = pyqtSignal(str)  # Signal emitted when workspace is loaded (path)
    save_error = pyqtSignal(str)  # Signal emitted when saving fails (error message)
    load_error = pyqtSignal(str)  # Signal emitted when loading fails (error message)

    def __init__(self, experiment_manager=None, node_controller=None):
        super().__init__()
        # References to experiment manager and node controller
        self.experiment_manager = experiment_manager or ExperimentManager()
        self.node_controller = node_controller

        # File format version (for future compatibility)
        self.file_format_version = "1.0"

    def set_node_controller(self, node_controller):
        """Sets the node controller for the serializer."""
        self.node_controller = node_controller

    def set_experiment_manager(self, experiment_manager):
        """Sets the experiment manager for the serializer."""
        self.experiment_manager = experiment_manager

    def save_workspace(self, filepath: str) -> bool:
        """
        Saves the workspace to a file.

        Args:
            filepath: Path to the save file

        Returns:
            bool: True on success, False on error
        """
        try:
            # Check if required dependencies are set
            if not self.experiment_manager or not self.node_controller:
                self.save_error.emit("Required dependencies not set")
                return False

            # Get data for serialization
            serialized_data = self._prepare_serialization_data()

            # Save data to file
            with open(filepath, 'wb') as file:
                pickle.dump(serialized_data, file, protocol=pickle.HIGHEST_PROTOCOL)

            # Notify about successful save
            self.workspace_saved.emit(filepath)
            return True

        except Exception as e:
            error_message = f"Error saving workspace: {str(e)}"
            print(error_message)
            self.save_error.emit(error_message)
            return False

    def load_workspace(self, filepath: str) -> bool:
        """
        Loads workspace from file.

        Args:
            filepath: Path to the file to load

        Returns:
            bool: True on success, False on error
        """
        try:
            # Check if required dependencies are set
            if not self.experiment_manager or not self.node_controller:
                self.load_error.emit("Required dependencies not set")
                return False

            # Check if file exists
            if not os.path.exists(filepath):
                self.load_error.emit(f"File {filepath} does not exist")
                return False

            # Load data from file
            with open(filepath, 'rb') as file:
                loaded_data = pickle.load(file)

            # Restore data
            self._restore_from_serialization_data(loaded_data)

            # Notify about successful load
            self.workspace_loaded.emit(filepath)
            return True

        except Exception as e:
            error_message = f"Error loading workspace: {str(e)}"
            print(error_message)
            self.load_error.emit(error_message)
            return False

    def _prepare_serialization_data(self) -> Dict:
        """
        Prepares data for serialization.

        Returns:
            Dict: Dictionary with serialization data
        """
        # Main dictionary for all data
        serialized_data = {
            "version": self.file_format_version,
            "experiments": {},
            "nodes": [],
            "edges": [],
            "node_positions": {},
            "experiment_node_map": {}
        }

        # Serialize experiments
        for exp_id, experiment in self.experiment_manager.experiments.items():
            # Create structure for storing all required experiment data
            exp_data = {
                "id": experiment.id,
                "name": experiment.name,
                "description": experiment.description,
                "task": experiment.task,
                "is_finished": experiment.is_finished,
                "train_time": experiment.train_time,
                "params": experiment._params,
                "input_data_params": vars(experiment.input_data_params),
                "parent_id": experiment.parent.id if experiment.parent else None,
                "type": "neural_network" if isinstance(experiment, GenericNeuralNetworkExperiment) else "standard"
            }

            # Add metrics data if experiment is finished
            if experiment.is_finished:
                exp_data["train_metrics"] = experiment.train_metrics
                exp_data["test_metrics"] = experiment.test_metrics

            # Save experiment data
            serialized_data["experiments"][exp_id] = exp_data

        # Serialize node data
        for node in self.node_controller.nodes:
            node_data = {
                "id": node.id,
                "name": node.get_name(),
                "position": (node.pos().x(), node.pos().y())
            }
            serialized_data["nodes"].append(node_data)

            # Save position separately for easy restoration
            serialized_data["node_positions"][node.id] = (node.pos().x(), node.pos().y())

            # Link node to experiment
            serialized_data["experiment_node_map"][node.id] = node.id  # In this case node ID = experiment ID

        # Serialize node connections
        for edge in self.node_controller.edges:
            edge_data = {
                "source_id": edge.source_node.id,
                "target_id": edge.target_node.id
            }
            serialized_data["edges"].append(edge_data)

        return serialized_data

    def _restore_from_serialization_data(self, data: Dict) -> None:
        """
        Restores workspace from serialized data.

        Args:
            data: Dictionary with serialized data
        """
        # Check file format version
        if "version" not in data or data["version"] != self.file_format_version:
            print(f"Warning: File format version ({data.get('version', 'unknown')}) "
                  f"differs from current ({self.file_format_version})")

        # Clear current workspace
        self._clear_current_workspace()

        # Restore nodes
        node_map = {}  # For storing mapping: old ID -> new node
        for node_data in data["nodes"]:
            # Create new node
            node = self.node_controller.create_node(
                x=node_data["position"][0],
                y=node_data["position"][1]
            )
            node.set_name(node_data["name"])

            # Save ID mapping
            node_map[node_data["id"]] = node

            # Explicitly set node position
            node.setPos(QPointF(node_data["position"][0], node_data["position"][1]))

        # Restore experiments
        experiment_map = {}  # old ID -> new experiment
        for exp_id, exp_data in data["experiments"].items():
            # Determine experiment type
            if exp_data["type"] == "neural_network":
                # Create neural network experiment
                experiment = GenericNeuralNetworkExperiment(
                    id=node_map[exp_id].id,  # Use new node ID
                    task=exp_data["task"],
                    model=None,  # Will be restored later
                    params=exp_data["params"]
                )
            else:
                # Create standard experiment
                experiment = Experiment(
                    id=node_map[exp_id].id,  # Use new node ID
                    task=exp_data["task"],
                    model=None,  # Will be restored later
                    params=exp_data["params"]
                )

            # Restore basic attributes
            experiment._name = exp_data["name"]
            experiment.description = exp_data["description"]
            experiment.is_finished = exp_data["is_finished"]
            experiment.train_time = exp_data["train_time"]

            # Restore input data parameters
            self._restore_input_data_params(experiment, exp_data["input_data_params"])

            # Restore metrics if experiment is finished
            if experiment.is_finished and "train_metrics" in exp_data:
                experiment.train_metrics = exp_data["train_metrics"]
                experiment.test_metrics = exp_data["test_metrics"]

            # Save experiment to manager
            self.experiment_manager.experiments[experiment.id] = experiment

            # Save ID mapping
            experiment_map[int(exp_id)] = experiment

        # Restore experiment parent relationships
        for exp_id, exp_data in data["experiments"].items():
            if exp_data["parent_id"] is not None:
                # Find corresponding parent experiment
                parent_exp = experiment_map.get(exp_data["parent_id"])
                if parent_exp:
                    # Set parent experiment
                    experiment_map[int(exp_id)].parent = parent_exp
                    # Add reference to child experiment in parent
                    parent_exp.children.append(experiment_map[int(exp_id)])

        # Restore node connections
        for edge_data in data["edges"]:
            # Find corresponding nodes
            source_node = node_map.get(edge_data["source_id"])
            target_node = node_map.get(edge_data["target_id"])

            if source_node and target_node:
                # Create connection
                self.node_controller.create_edge(source_node, target_node)

        # Update scene to display all elements correctly
        if self.node_controller.scene:
            self.node_controller.scene.update()

    def _restore_input_data_params(self, experiment, params_data):
        """
        Restores input data parameters for experiment.

        Args:
            experiment: Experiment to restore parameters for
            params_data: Dictionary with parameters
        """
        # Determine parameter type based on experiment type
        if isinstance(experiment, GenericNeuralNetworkExperiment):
            params = experiment.input_data_params
        else:
            params = experiment.input_data_params

        # Restore all attributes
        for key, value in params_data.items():
            if hasattr(params, key):
                setattr(params, key, value)

    def _clear_current_workspace(self):
        """Clears current workspace"""
        # Clear experiments
        self.experiment_manager.experiments = {}

        # Clear nodes and connections from scene
        for node in list(self.node_controller.nodes):
            self.node_controller.delete_node(node)

        # Additional cleanup if needed
        self.node_controller.nodes = []
        self.node_controller.edges = []