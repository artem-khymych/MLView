from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QListWidgetItem, QListWidget, QLabel, QVBoxLayout, QWidget



class NodeInspector(QWidget):
    """Nodes manager - creates and manages nodes"""

    def __init__(self, node_controller):
        super().__init__()
        self.node_controller = node_controller
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        label = QLabel("Інспектор Експериментів")
        label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(label)

        self.nodes_list = QListWidget()
        self.nodes_list.itemClicked.connect(self.on_item_clicked)
        layout.addWidget(self.nodes_list)

        self.node_controller.node_created.connect(self.add_node_to_inspector)
        self.node_controller.node_deleted.connect(self.remove_node_from_inspector)
        self.node_controller.node_renamed.connect(self.update_node_in_inspector)

    def add_node_to_inspector(self, node):
        item = QListWidgetItem(node.get_name())
        item.setData(Qt.UserRole, node.id)  # Зберігаємо ID вузла
        self.nodes_list.addItem(item)

    def remove_node_from_inspector(self, node_id):
        for i in range(self.nodes_list.count()):
            item = self.nodes_list.item(i)
            if item.data(Qt.UserRole) == node_id:
                self.nodes_list.takeItem(i)
                break

    def update_node_in_inspector(self, node):
        for i in range(self.nodes_list.count()):
            item = self.nodes_list.item(i)
            if item.data(Qt.UserRole) == node.id:
                item.setText(node.get_name())
                break

    def on_item_clicked(self, item):
        node_id = item.data(Qt.UserRole)
        self.node_controller.center_on_node(node_id)

