from typing import Dict, List, Tuple, Any, Optional
from PyQt5.QtCore import QObject, pyqtSlot, Qt
from PyQt5.QtWidgets import QGraphicsView, QWidget, QMessageBox, QFileDialog

from project.controllers.inspector_controller import InspectorController
from project.controllers.node_controller import NodeController
from project.controllers.serializer import WorkspaceSerializer
from project.logic.experiment_manager import ExperimentManager

class WorkspaceManager(QObject):
    """
    Workspace manager responsible for saving/loading program state
    and managing project files.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # Path to current project file
        self.current_project_path: Optional[str] = None

        # Project modification status (unsaved changes)
        self.has_unsaved_changes: bool = False

        # Serializer for save/load operations
        self.serializer = WorkspaceSerializer()

        # References to other system components
        self.experiment_manager: Optional[ExperimentManager] = None
        self.node_controller: Optional[NodeController] = None
        self.work_area: Optional[QGraphicsView] = None  # Reference to work area

        # Connect serializer signals
        self._connect_serializer_signals()

    def set_inspector_controller(self, inspector_controller: InspectorController):
        self.inspector_controller = inspector_controller

    def set_experiment_manager(self, manager: ExperimentManager):
        """Sets experiment manager for WorkspaceManager."""
        self.experiment_manager = manager
        self.serializer.set_experiment_manager(manager)

    def set_node_controller(self, controller: NodeController):
        """Sets node controller for WorkspaceManager."""
        self.node_controller = controller
        self.serializer.set_node_controller(controller)

        # Connect controller signals to track changes
        self._connect_node_controller_signals()

    def set_work_area(self, work_area: QGraphicsView):
        """Sets work area for WorkspaceManager."""
        self.work_area = work_area

    def _connect_serializer_signals(self):
        """Connects serializer signals to corresponding slots."""
        self.serializer.workspace_saved.connect(self._on_workspace_saved)
        self.serializer.workspace_loaded.connect(self._on_workspace_loaded)
        self.serializer.save_error.connect(self._on_save_error)
        self.serializer.load_error.connect(self._on_load_error)

    def _connect_node_controller_signals(self):
        """Connects node controller signals to track changes."""
        if self.node_controller:
            # Track graph changes
            self.node_controller.node_created.connect(self._on_workspace_modified)
            self.node_controller.node_deleted.connect(self._on_workspace_modified)
            self.node_controller.node_renamed.connect(self._on_workspace_modified)
            self.node_controller.experiment_inherited.connect(self._on_workspace_modified)

    @pyqtSlot()
    def _on_workspace_modified(self):
        """Handler for workspace modification events."""
        self.has_unsaved_changes = True

    @pyqtSlot(str)
    def _on_workspace_saved(self, path: str):
        """Handler for successful workspace save."""
        self.current_project_path = path
        self.has_unsaved_changes = False
        print(f"Workspace successfully saved: {path}")

    @pyqtSlot(str)
    def _on_workspace_loaded(self, path: str):
        """Handler for successful workspace load."""
        self.current_project_path = path
        self.has_unsaved_changes = False
        print(f"Workspace successfully loaded: {path}")
        for node in self.node_controller.nodes:
            self.inspector_controller.update_node_in_inspector(node)
        # Fit view to content after loading
        self.fit_view_to_content()

    @pyqtSlot(str)
    def _on_save_error(self, error_msg: str):
        """Handler for save errors."""
        print(f"Save error: {error_msg}")
        QMessageBox.critical(None, "Помилка збереження", error_msg)

    @pyqtSlot(str)
    def _on_load_error(self, error_msg: str):
        """Handler for load errors."""
        print(f"Load error: {error_msg}")
        QMessageBox.critical(None, "Помилка завантаження", error_msg)

    def new_project(self):
        """Creates new project."""
        # Check for unsaved changes
        if self.has_unsaved_changes and not self._confirm_discard_changes():
            return False

        # Clear current data
        self._clear_workspace()

        # Reset project path
        self.current_project_path = None
        self.has_unsaved_changes = False

        return True

    def save_project(self, parent_widget: Optional[QWidget] = None) -> bool:
        """
        Saves current project.

        Args:
            parent_widget: Parent widget for dialogs

        Returns:
            bool: True on success, False on error
        """
        # Check if project path exists
        if not self.current_project_path:
            return self.save_project_as(parent_widget)

        # Save project to current path
        success = self.serializer.save_workspace(self.current_project_path)
        return success

    def save_project_as(self, parent_widget: Optional[QWidget] = None) -> bool:
        """
        Saves project to new file.

        Args:
            parent_widget: Parent widget for dialogs

        Returns:
            bool: True on success, False on error
        """
        # Open file save dialog
        options = QFileDialog.Options()
        filepath, _ = QFileDialog.getSaveFileName(
            parent_widget,
            "Зберегти проект",
            "",
            "ML Project Files (*.mlproj);;All Files (*)",
            options=options
        )

        if filepath:
            # Add extension if not specified
            if not filepath.endswith('.mlproj'):
                filepath += '.mlproj'

            # Save project to selected file
            return self.serializer.save_workspace(filepath)

        return False

    def open_project(self, parent_widget: Optional[QWidget] = None) -> bool:
        """
        Opens existing project.

        Args:
            parent_widget: Parent widget for dialogs

        Returns:
            bool: True on success, False on error
        """
        # Check for unsaved changes
        if self.has_unsaved_changes and not self._confirm_discard_changes():
            return False

        # Open file load dialog
        options = QFileDialog.Options()
        filepath, _ = QFileDialog.getOpenFileName(
            parent_widget,
            "Відкрити проект",
            "",
            "ML Project Files (*.mlproj);;All Files (*)",
            options=options
        )

        if filepath:
            # Load project from selected file
            return self.serializer.load_workspace(filepath)

        return False

    def fit_view_to_content(self):
        """Fits view to all scene items."""
        if self.work_area and self.node_controller and self.node_controller.scene:
            # Do nothing if no items exist
            if not self.node_controller.nodes:
                return

            # Calculate bounding rectangle for all nodes
            rect = None
            for node in self.node_controller.nodes:
                node_rect = node.sceneBoundingRect()
                if rect is None:
                    rect = node_rect
                else:
                    rect = rect.united(node_rect)

            # Add margin around items
            if rect:
                margin = 50  # Pixel margin
                rect = rect.adjusted(-margin, -margin, margin, margin)

                # Fit view to this rectangle
                self.work_area.fitInView(rect, Qt.KeepAspectRatio)

    def _confirm_discard_changes(self) -> bool:
        """
        Confirms discarding unsaved changes with user.

        Returns:
            bool: True if user confirmed discard, False otherwise
        """
        reply = QMessageBox.question(
            None,
            "Незбережені зміни",
            "Є незбережені зміни. Бажаєте зберегти їх перед продовженням?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Save
        )

        if reply == QMessageBox.Save:
            # Save changes
            return self.save_project()
        elif reply == QMessageBox.Discard:
            # Discard changes
            return True
        else:  # QMessageBox.Cancel
            # Cancel action
            return False

    def _clear_workspace(self):
        """Clears workspace."""
        # Clear experiments in manager
        if self.experiment_manager:
            self.experiment_manager.experiments = {}

        if self.inspector_controller:
            for node in list(self.node_controller.nodes):
                self.inspector_controller.remove_node_from_inspector(node.id)

        # Clear nodes on scene
        if self.node_controller:
            for node in list(self.node_controller.nodes):
                self.node_controller.delete_node(node)

