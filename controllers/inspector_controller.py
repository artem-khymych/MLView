from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QListWidgetItem, QWidget, QListWidget

from project.controllers.basic_contoller import BasicController
from project.controllers.node_controller import NodeController


class InspectorController(BasicController):
    """Node inspector - displays a list of nodes and allows managing them."""

    def __init__(self, view, scene, node_view):
        super().__init__(view)
        self.node_controller = NodeController(scene, node_view)
        self.nodes_list = self.view.findChild(QListWidget, "nodes_list")

        # Connecting to the node controller
        self.node_controller.node_created.connect(self.add_node_to_inspector)
        self.node_controller.node_deleted.connect(self.remove_node_from_inspector)
        self.node_controller.node_renamed.connect(self.update_node_in_inspector)
        self.nodes_list.itemClicked.connect(self.on_item_clicked)

    def add_node_to_inspector(self, node):
        """Adds a node to the inspector."""
        item = QListWidgetItem(node.get_name())
        item.setData(Qt.UserRole, node.id)  # Storing the node ID
        self.nodes_list.addItem(item)

    def remove_node_from_inspector(self, node_id):
        """Removes a node from the inspector by ID."""
        for i in range(self.nodes_list.count()):
            item = self.nodes_list.item(i)
            if item.data(Qt.UserRole) == node_id:
                self.nodes_list.takeItem(i)
                break

    def update_node_in_inspector(self, node):
        """Updates the node display in the inspector."""
        for i in range(self.nodes_list.count()):
            item = self.nodes_list.item(i)
            if item.data(Qt.UserRole) == node.id:
                item.setText(node.get_name())
                self.node_controller.update_experiment_name.emit(node.id, node.get_name())
                break

    def on_item_clicked(self, item):
        """Click handler for list items."""
        node_id = item.data(Qt.UserRole)
        self.node_controller.center_on_node(node_id)