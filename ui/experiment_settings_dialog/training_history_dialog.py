import sys
from PyQt5.QtWidgets import (QApplication, QDialog, QVBoxLayout, QHBoxLayout,
                             QTabWidget, QLabel, QTableWidget, QTableWidgetItem,
                             QSplitter, QWidget, QPushButton)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np


class TrainingHistoryDialog(QDialog):
    def __init__(self, parent=None):
        super(TrainingHistoryDialog, self).__init__(parent)
        self.setWindowTitle("Історія навчання нейромережі")
        self.resize(900, 600)

        main_layout = QVBoxLayout(self)

        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        self.plots_tab = QWidget()
        self.plots_layout = QVBoxLayout(self.plots_tab)
        self.tab_widget.addTab(self.plots_tab, "Графіки")

        self.table_tab = QWidget()
        self.table_layout = QVBoxLayout(self.table_tab)
        self.tab_widget.addTab(self.table_tab, "Таблиця значень")

        self.stats_tab = QWidget()
        self.stats_layout = QVBoxLayout(self.stats_tab)
        self.tab_widget.addTab(self.stats_tab, "Статистика")

        self.close_button = QPushButton("Закрити")
        self.close_button.clicked.connect(self.close)
        main_layout.addWidget(self.close_button)

    def show_history(self, history):
        """

        :param history: object history.history
        """
        self._clear_layouts()

        if not isinstance(history, dict):
            try:
                history = history.history
            except AttributeError:
                error_label = QLabel("Помилка: Переданий об'єкт не є допустимою історією навчання.")
                self.plots_layout.addWidget(error_label)
                return

        self._create_plots(history)

        self._create_table(history)

        self._create_stats(history)

        self.show()
        self.exec_()

    def _clear_layouts(self):
        for layout in [self.plots_layout, self.table_layout, self.stats_layout]:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()

    def _create_plots(self, history):
        # Визначення метрик для візуалізації
        metrics = [key for key in history.keys() if not key.startswith('val_')]
        val_metrics = [key for key in history.keys() if key.startswith('val_')]

        plot_container = QWidget()
        plot_container_layout = QVBoxLayout(plot_container)

        for metric in metrics:
            val_metric = f'val_{metric}' if f'val_{metric}' in val_metrics else None

            plot_widget = QWidget()
            plot_layout = QVBoxLayout(plot_widget)

            fig = Figure(figsize=(8, 4), dpi=100)
            ax = fig.add_subplot(111)

            canvas = FigureCanvas(fig)

            epochs = range(1, len(history[metric]) + 1)
            ax.plot(epochs, history[metric], 'b-', label=f'Тренування ({metric})')

            if val_metric:
                ax.plot(epochs, history[val_metric], 'r-', label=f'Валідація ({val_metric})')

            ax.set_title(f'Метрика: {metric}')
            ax.set_xlabel('Епохи')
            ax.set_ylabel('Значення')
            ax.legend()
            ax.grid(True)

            plot_layout.addWidget(canvas)

            plot_container_layout.addWidget(plot_widget)

        scroll_area = QWidget()
        scroll_layout = QVBoxLayout(scroll_area)
        scroll_layout.addWidget(plot_container)
        self.plots_layout.addWidget(scroll_area)

    def _create_table(self, history):
        num_epochs = len(list(history.values())[0]) if history else 0
        metrics = list(history.keys())

        table = QTableWidget(num_epochs, len(metrics))
        table.setHorizontalHeaderLabels(metrics)

        for col, metric in enumerate(metrics):
            for row in range(num_epochs):
                value = history[metric][row]
                item = QTableWidgetItem(f"{value:.6f}")
                table.setItem(row, col, item)

        table.setVerticalHeaderLabels([f"Епоха {i + 1}" for i in range(num_epochs)])

        self.table_layout.addWidget(table)

    def _create_stats(self, history):
        metrics = list(history.keys())

        stats_table = QTableWidget(5, len(metrics))
        stats_table.setHorizontalHeaderLabels(metrics)
        stats_table.setVerticalHeaderLabels(["Мінімум", "Максимум", "Середнє", "Останнє", "Різниця макс-мін"])

        for col, metric in enumerate(metrics):
            values = history[metric]

            min_val = min(values)
            max_val = max(values)
            avg_val = sum(values) / len(values)
            last_val = values[-1]
            diff_val = max_val - min_val

            stats = [min_val, max_val, avg_val, last_val, diff_val]
            for row, stat in enumerate(stats):
                item = QTableWidgetItem(f"{stat:.6f}")
                stats_table.setItem(row, col, item)

        self.stats_layout.addWidget(stats_table)

        info_text = "Загальна інформація:\n"
        info_text += f"- Кількість епох: {len(list(history.values())[0])}\n"
        info_text += f"- Кількість метрик: {len(metrics)}\n"

        train_metrics = [m for m in metrics if not m.startswith('val_')]
        val_metrics = [m for m in metrics if m.startswith('val_')]

        info_text += f"- Тренувальні метрики: {', '.join(train_metrics)}\n"
        info_text += f"- Валідаційні метрики: {', '.join(val_metrics)}\n"

        for metric in train_metrics:
            values = history[metric]
            improvement = values[-1] - values[0]
            direction = "↑" if improvement > 0 else "↓"

            is_better = (improvement < 0 and "loss" in metric.lower()) or (
                        improvement > 0 and "loss" not in metric.lower())
            performance = "покращилась" if is_better else "погіршилась"

            info_text += f"- Метрика '{metric}' {performance} на {abs(improvement):.6f} {direction}\n"

        info_label = QLabel(info_text)
        info_label.setAlignment(Qt.AlignTop)
        self.stats_layout.addWidget(info_label)
