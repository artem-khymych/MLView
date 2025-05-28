from PyQt5.QtWidgets import QVBoxLayout, QTableWidget, QHeaderView, QTableWidgetItem, QWidget, QPushButton


class MetricsTabWidget(QWidget):
    """Metrics tabs of the model"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        self.layout = QVBoxLayout(self)

        self.table = QTableWidget(1, 3)
        self.table.setHorizontalHeaderLabels(["Метрика", "Тренувальні значення", "Навчальні значення"])
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self.compare_button = QPushButton("Порівняти гілку")
        self.compare_all = QPushButton("Порівняти всі")
        self.layout.addWidget(self.table)
        self.layout.addWidget(self.compare_button)
        self.layout.addWidget(self.compare_all)

    def update_metrics(self, metrics_data: dict):
        """

        Args:
            metrics_data: {metric_name: {'train': value, 'test': value}}
        """
        self.table.clearContents()

        self.table.setRowCount(len(metrics_data))

        for row, (metric_name, values) in enumerate(metrics_data.items()):
            self.table.setItem(row, 0, QTableWidgetItem(metric_name))

            if values['train'] is not None:
                self.table.setItem(row, 1, QTableWidgetItem(f"{values['train']:.4f}"))
            else:
                self.table.setItem(row, 1, QTableWidgetItem("N/A"))

            if values['test'] is not None:
                self.table.setItem(row, 2, QTableWidgetItem(f"{values['test']:.4f}"))
            else:
                self.table.setItem(row, 2, QTableWidgetItem("N/A"))

        self.table.setEnabled(len(metrics_data) > 0)