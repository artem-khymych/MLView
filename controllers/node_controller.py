from PyQt5.QtCore import Qt, QPoint, QTimer
from PyQt5.QtWidgets import QGraphicsView, QMenu, QAction, QMessageBox, QGraphicsItem

from project.ui.Edge import Edge
from project.ui.node import Node
from PyQt5.QtCore import pyqtSignal, QObject


class NodeController:
    """Node controller - responsible for creating, managing and interacting with nodes."""

    def __init__(self, scene, view):
        self.scene = scene
        self.view = view
        self.nodes = []
        self.edges = []  # Adding a list to store connections
        self.active_node = None
        self.drag_timer = QTimer()
        self.drag_timer.setSingleShot(True)
        self.drag_timer.timeout.connect(self._start_dragging)
        self.current_press_node = None

        # Create a mediator class for signals
        class SignalEmitter(QObject):
            node_created = pyqtSignal(object)  # Node creation signal
            node_deleted = pyqtSignal(int)  # Node deletion signal (passes ID)
            node_renamed = pyqtSignal(object)  # Node rename signal
            update_experiment_name = pyqtSignal(int, str)
            nodeInfoOpened = pyqtSignal(int)  # Signal for opening node info
            experiment_inherited = pyqtSignal(int)  # Experiment inheritance signal

        self.signals = SignalEmitter()
        self.node_created = self.signals.node_created
        self.node_deleted = self.signals.node_deleted
        self.node_renamed = self.signals.node_renamed
        self.nodeInfoOpened = self.signals.nodeInfoOpened
        self.experiment_inherited = self.signals.experiment_inherited
        self.update_experiment_name = self.signals.update_experiment_name

        self.scene.installEventFilter(view)
        view.mousePressEvent = self._view_mouse_press
        view.mouseReleaseEvent = self._view_mouse_release
        view.mouseMoveEvent = self._view_mouse_move
        view.contextMenuEvent = self._view_context_menu

    def create_node(self, x=None, y=None):
        """Creates a new node at specified coordinates or at the center of visible area."""
        node = Node()

        if x is None or y is None:
            # Get the visible viewport area
            viewport_rect = self.view.viewport().rect()

            # Convert viewport center to scene coordinates
            viewport_center = QPoint(viewport_rect.width() // 2, viewport_rect.height() // 2)
            scene_center = self.view.mapToScene(viewport_center)

            # Account for node size for precise centering
            node_width = node.boundingRect().width()
            node_height = node.boundingRect().height()

            # Set node position at viewport center
            node.setPos(scene_center.x() - node_width / 2, scene_center.y() - node_height / 2)
        else:
            # Set node position at specified coordinates
            node.setPos(x, y)

        self.scene.addItem(node)
        self.nodes.append(node)

        self.node_created.emit(node)

        return node

    def create_inherited_node(self, parent_node_id):
        """Creates a new node that inherits from the node with specified ID."""
        # Find parent node by ID
        parent_node = self.find_node_by_id(parent_node_id)

        if not parent_node:
            print(f"Error: node with ID {parent_node_id} not found")
            return None

        # Determine position for new node (slightly to the right and below parent)
        parent_pos = parent_node.pos()
        new_x = parent_pos.x() + 150  # Right offset
        new_y = parent_pos.y() + 100  # Down offset

        # Create new node
        new_node = self.create_node(new_x, new_y)

        # Set name indicating inheritance
        new_node.set_name(f"Успадковано від {parent_node.get_name()}")

        # Create connection between nodes
        self.create_edge(parent_node, new_node)

        return new_node

    def create_edge(self, source_node, target_node):
        """Creates a connection between two nodes."""
        edge = Edge(source_node, target_node)
        self.scene.addItem(edge)
        self.edges.append(edge)

        # Connect edge update when nodes move
        source_node.itemChange = self._wrap_item_change(source_node, edge)
        target_node.itemChange = self._wrap_item_change(target_node, edge)

        return edge

    def _wrap_item_change(self, node, edge):
        """Creates a wrapper for node's itemChange method to update the connection."""
        original_item_change = node.itemChange if hasattr(node, 'itemChange') else lambda change, value: value

        def wrapped_item_change(change, value):
            result = original_item_change(change, value)
            if change == QGraphicsItem.ItemPositionChange or change == QGraphicsItem.ItemPositionHasChanged:
                edge.update_position()
            return result

        return wrapped_item_change

    def find_node_by_id(self, node_id):
        """Finds a node by its ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def delete_node(self, node):
        """Deletes a node and all its connections."""
        if node in self.nodes:
            node_id = node.id

            # Remove all connections related to this node
            edges_to_remove = [edge for edge in self.edges
                               if edge.source_node == node or edge.target_node == node]

            for edge in edges_to_remove:
                self.scene.removeItem(edge)
                self.edges.remove(edge)

            # Remove the node itself
            self.scene.removeItem(node)
            self.nodes.remove(node)

            if self.active_node == node:
                self.active_node = None

            # Emit node deletion signal
            self.node_deleted.emit(node_id)

    def open_node_info(self, node):
        """Opens a dialog with node information."""
        self.nodeInfoOpened.emit(node.id)

    def edit_node_name(self, node):
        """Activates node name editing."""
        node.start_editing_name()
        # Subscribe to editing finished signal
        node.name_editor.editingFinished.connect(lambda: self.node_renamed.emit(node))

    def center_on_node(self, node_id):
        """Centers view on the node with specified ID."""
        for node in self.nodes:
            if node.id == node_id:
                # Get node center
                node_center = node.mapToScene(
                    node.boundingRect().center().x(),
                    node.boundingRect().center().y()
                )
                # Center view on node
                self.view.centerOn(node_center)
                break

    def _view_mouse_press(self, event):
        """Mouse press event handler for GraphicsView."""
        # Get item under mouse
        pos = self.view.mapToScene(event.pos())
        item = self.scene.itemAt(pos, self.view.transform())

        if event.button() == Qt.LeftButton and isinstance(item, Node):
            self.current_press_node = item
            # Start timer to detect click-and-hold
            self.drag_timer.start(200)
        else:
            # Standard processing for non-node items
            QGraphicsView.mousePressEvent(self.view, event)

    def _view_mouse_release(self, event):
        """Mouse release event handler for GraphicsView."""
        if event.button() == Qt.LeftButton and self.current_press_node:
            if self.drag_timer.isActive():
                # If timer is still active, it's a click
                self.drag_timer.stop()
                self.open_node_info(self.current_press_node)
            else:
                # If timer is inactive, it's drag end
                if self.active_node:
                    self.active_node.set_active(False)
                    self.active_node = None

            self.current_press_node = None
        else:
            # Standard processing for other cases
            QGraphicsView.mouseReleaseEvent(self.view, event)

    def _view_mouse_move(self, event):
        """Mouse move event handler for GraphicsView."""
        if self.active_node:
            # If there's an active node, update its position
            pos = self.view.mapToScene(event.pos())
            node_width = self.active_node.boundingRect().width()
            node_height = self.active_node.boundingRect().height()
            self.active_node.setPos(pos.x() - node_width / 2, pos.y() - node_height / 2)
        else:
            # Standard processing for other cases
            QGraphicsView.mouseMoveEvent(self.view, event)

    def _view_context_menu(self, event):
        """Context menu event handler."""
        # Get item under mouse
        pos = self.view.mapToScene(event.pos())
        item = self.scene.itemAt(pos, self.view.transform())

        if isinstance(item, Node):
            # Create context menu
            context_menu = QMenu(self.view)

            # Add actions
            rename_action = QAction("Перейменувати", self.view)
            delete_action = QAction("Видалити", self.view)

            # Connect handlers
            rename_action.triggered.connect(lambda: self.edit_node_name(item))
            delete_action.triggered.connect(lambda: self.delete_node(item))

            # Add actions to menu
            context_menu.addAction(rename_action)
            context_menu.addAction(delete_action)

            # Show menu
            context_menu.exec_(event.globalPos())
        else:
            # Standard processing for other cases
            QGraphicsView.contextMenuEvent(self.view, event)

    def _start_dragging(self):
        """Activates node dragging mode."""
        if self.current_press_node:
            self.active_node = self.current_press_node
            self.active_node.set_active(True)