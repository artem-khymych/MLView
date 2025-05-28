from PyQt5.QtCore import QRectF, QLineF
from PyQt5.QtGui import QColor, QPen
from PyQt5.QtWidgets import QGraphicsItem


class Edge(QGraphicsItem):
    """Class representing a connection between nodes."""

    def __init__(self, source_node, target_node, parent=None):
        super().__init__(parent)
        self.source_node = source_node
        self.target_node = target_node
        self.setZValue(-1)  # Place the connection below nodes

        # Connect to node position changes
        self.source_node.setFlag(QGraphicsItem.ItemSendsScenePositionChanges)
        self.target_node.setFlag(QGraphicsItem.ItemSendsScenePositionChanges)

        self.update_position()

    def update_position(self):
        """Updates line position when nodes are moved."""
        self.prepareGeometryChange()
        self.update()

    def boundingRect(self):
        """Defines connection boundaries for rendering optimization."""
        if not self.source_node or not self.target_node:
            return QRectF()

        # Get node centers
        source_center = self.source_node.mapToScene(
            self.source_node.boundingRect().center().x(),
            self.source_node.boundingRect().center().y()
        )
        target_center = self.target_node.mapToScene(
            self.target_node.boundingRect().center().x(),
            self.target_node.boundingRect().center().y()
        )

        # Create a rectangle enclosing both centers with small padding
        return QRectF(
            min(source_center.x(), target_center.x()) - 5,
            min(source_center.y(), target_center.y()) - 5,
            abs(source_center.x() - target_center.x()) + 10,
            abs(source_center.y() - target_center.y()) + 10
        )

    def paint(self, painter, option, widget=None):
        """Draws a line between nodes."""
        if not self.source_node or not self.target_node:
            return

        # Get node centers
        source_center = self.source_node.mapToScene(
            self.source_node.boundingRect().center().x(),
            self.source_node.boundingRect().center().y()
        )
        target_center = self.target_node.mapToScene(
            self.target_node.boundingRect().center().x(),
            self.target_node.boundingRect().center().y()
        )

        # Configure pen for drawing
        pen = QPen(QColor(0, 0, 0))  # Чорний колір (Black color)
        pen.setWidth(2)
        painter.setPen(pen)

        # Draw the line
        line = QLineF(source_center, target_center)
        painter.drawLine(line)