from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QHeaderView, QTableWidgetItem, QTableWidget,
                             QPushButton, QVBoxLayout, QDialog, QLabel,
                             QHBoxLayout, QFrame)
from PyQt5.QtGui import QFont, QColor, QBrush

from project.logic.evaluation.task_register import TaskType
from project.logic.experiment.generic_nn_experiment import GenericNeuralNetworkExperiment


class ExperimentComparisonDialog(QDialog):
    """
    Dialog window for comparing metrics of different experiments.
    """

    def __init__(self, experiments):
        super().__init__()
        self.experiments = experiments
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Порівняння експериментів")
        self.setMinimumSize(1000, 900)

        main_layout = QVBoxLayout()

        # Create table for training metrics with header
        train_layout = QVBoxLayout()
        train_label = QLabel("Тренувальні метрики")
        train_label.setFont(QFont("Arial", 12, QFont.Bold))
        train_layout.addWidget(train_label)

        self.train_table = self.create_metric_table("train_metrics")
        train_layout.addWidget(self.train_table)

        # Add separator line
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)

        # Create table for test metrics with header
        test_layout = QVBoxLayout()
        test_label = QLabel("Тестові метрики")
        test_label.setFont(QFont("Arial", 12, QFont.Bold))
        test_layout.addWidget(test_label)

        self.test_table = self.create_metric_table("test_metrics")
        test_layout.addWidget(self.test_table)

        # Close button
        button_layout = QHBoxLayout()
        close_button = QPushButton("Закрити")
        close_button.setMinimumHeight(30)
        close_button.clicked.connect(self.accept)
        button_layout.addStretch()
        button_layout.addWidget(close_button)

        # Add all elements to main layout
        main_layout.addLayout(train_layout)
        main_layout.addWidget(line)
        main_layout.addLayout(test_layout)
        main_layout.addLayout(button_layout)

        # Add legend
        legend_layout = QHBoxLayout()

        best_indicator = QFrame()
        best_indicator.setFixedSize(20, 20)
        best_indicator.setAttribute(Qt.WA_StyledBackground, True)
        best_indicator.setStyleSheet("background-color: #B4FFB4; border: 1px solid #2ECC71;")

        worst_indicator = QFrame()
        worst_indicator.setFixedSize(20, 20)
        worst_indicator.setAttribute(Qt.WA_StyledBackground, True)
        worst_indicator.setStyleSheet("background-color: #FFB4B4; border: 1px solid #E74C3C;")

        legend_layout.addWidget(best_indicator)
        legend_layout.addWidget(QLabel("Найкраще значення"))
        legend_layout.addSpacing(20)
        legend_layout.addWidget(worst_indicator)
        legend_layout.addWidget(QLabel("Найгірше значення"))
        legend_layout.addStretch()

        main_layout.addLayout(legend_layout)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

    def create_metric_table(self, metrics_attr):
        """
        Creates a table for displaying experiment metrics.

        Args:
            metrics_attr (str): Name of the metrics attribute ('train_metrics' or 'test_metrics')

        Returns:
            QTableWidget: Table with metrics
        """
        # Get information about metrics (maximize or minimize)
        meta_info = {}
        if self.experiments and hasattr(self.experiments[0], 'metric_strategy'):
            meta_info = self.experiments[0].metric_strategy.get_metainformation()

        # Collect all unique metrics from all experiments
        all_metrics = set()
        for exp in self.experiments:
            if hasattr(exp, metrics_attr) and getattr(exp, metrics_attr):
                all_metrics.update(getattr(exp, metrics_attr).keys())

        # Separate metrics into comparable and non-comparable
        comparable_metrics = []
        non_comparable_metrics = []

        for metric in all_metrics:
            if metric in meta_info and meta_info[metric] is not None:
                comparable_metrics.append(metric)
            else:
                non_comparable_metrics.append(metric)

        # Sort metrics for consistent display
        comparable_metrics = sorted(comparable_metrics)
        non_comparable_metrics = sorted(non_comparable_metrics)

        # Create table
        table = QTableWidget()
        total_metrics = len(comparable_metrics) + len(non_comparable_metrics) + 1  # +1 for general information
        if non_comparable_metrics:
            total_metrics += 1  # Add row for separator

        table.setRowCount(total_metrics)
        table.setColumnCount(len(self.experiments) + 1)  # +1 for row headers

        # Set clear font for table
        table_font = QFont("Arial", 10)
        table.setFont(table_font)

        # Table header - explicitly set font and make bold
        header_font = QFont("Arial", 10, QFont.Bold)

        # Column names
        header_item = QTableWidgetItem("Метрика")
        header_item.setFont(header_font)
        table.setHorizontalHeaderItem(0, header_item)

        for i, exp in enumerate(self.experiments):
            header_text = f"{exp.name} (ID: {exp.id})"
            header_item = QTableWidgetItem(header_text)
            header_item.setFont(header_font)
            table.setHorizontalHeaderItem(i + 1, header_item)

        # Add general information about experiments
        info_item = QTableWidgetItem("Тип завдання")
        info_item.setFont(header_font)
        table.setItem(0, 0, info_item)

        for i, exp in enumerate(self.experiments):
            task_item = QTableWidgetItem(exp.task.value if isinstance(exp, GenericNeuralNetworkExperiment) else TaskType(exp.task).value)
            table.setItem(0, i + 1, task_item)

        # Add comparable metrics and highlight best/worst values
        for row, metric in enumerate(comparable_metrics):
            metric_item = QTableWidgetItem(metric)
            metric_item.setFont(table_font)
            table.setItem(row + 1, 0, metric_item)

            # Collect all metric values for comparison
            values = []
            indices = []
            for i, exp in enumerate(self.experiments):
                metrics = getattr(exp, metrics_attr, {})
                if metric in metrics:
                    value = metrics[metric]
                    if isinstance(value, (int, float)):
                        values.append(value)
                        indices.append(i)
                    else:
                        values.append(None)
                        indices.append(i)
                else:
                    values.append(None)
                    indices.append(i)

            # Find best and worst values
            best_index = None
            worst_index = None
            valid_values = [v for v in values if v is not None]

            if valid_values:
                if meta_info.get(metric, True):  # True by default, higher is better
                    max_val = max([v for v in values if v is not None], default=None)
                    min_val = min([v for v in values if v is not None], default=None)
                    best_index = values.index(max_val) if max_val is not None else None
                    worst_index = values.index(min_val) if min_val is not None else None
                else:  # False, lower is better
                    min_val = min([v for v in values if v is not None], default=None)
                    max_val = max([v for v in values if v is not None], default=None)
                    best_index = values.index(min_val) if min_val is not None else None
                    worst_index = values.index(max_val) if max_val is not None else None

            # Add metric values
            for i, exp in enumerate(self.experiments):
                metrics = getattr(exp, metrics_attr, {})
                if metric in metrics:
                    value = metrics[metric]
                    # Format numeric values for better display
                    if isinstance(value, (int, float)):
                        value_str = f"{value:.4f}" if isinstance(value, float) else str(value)
                    else:
                        value_str = str(value)
                    value_item = QTableWidgetItem(value_str)

                    # Highlight best value with green
                    if i == best_index:
                        value_item.setBackground(QBrush(QColor(180, 255, 180)))  # Light green
                    # Highlight worst value with red
                    elif i == worst_index:
                        value_item.setBackground(QBrush(QColor(255, 180, 180)))  # Light red

                    table.setItem(row + 1, i + 1, value_item)
                else:
                    na_item = QTableWidgetItem("N/A")
                    table.setItem(row + 1, i + 1, na_item)

        # Add separator if there are non-comparable metrics
        current_row = len(comparable_metrics) + 1

        if non_comparable_metrics:
            # Add separator
            for col in range(table.columnCount()):
                separator_item = QTableWidgetItem("")
                separator_item.setBackground(QBrush(QColor(200, 200, 200)))  # Gray color for separator
                table.setItem(current_row, col, separator_item)

            # Set greater height for separator row
            table.setRowHeight(current_row, 5)
            current_row += 1

            # Add header for non-comparable metrics
            header_item = QTableWidgetItem("Непорівнювані метрики")
            header_item.setFont(header_font)
            table.setItem(current_row, 0, header_item)
            current_row += 1

            # Add non-comparable metrics
            for row_offset, metric in enumerate(non_comparable_metrics):
                row = current_row + row_offset
                metric_item = QTableWidgetItem(metric)
                metric_item.setFont(table_font)
                table.setItem(row, 0, metric_item)

                for i, exp in enumerate(self.experiments):
                    metrics = getattr(exp, metrics_attr, {})
                    if metric in metrics:
                        value = metrics[metric]
                        if isinstance(value, (int, float)):
                            value_str = f"{value:.4f}" if isinstance(value, float) else str(value)
                        else:
                            value_str = str(value)
                        value_item = QTableWidgetItem(value_str)
                        table.setItem(row, i + 1, value_item)
                    else:
                        na_item = QTableWidgetItem("N/A")
                        table.setItem(row, i + 1, na_item)

        # Configure table appearance
        table.setAlternatingRowColors(True)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.verticalHeader().setVisible(False)
        table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Ensure header visibility
        table.horizontalHeader().setVisible(True)

        return table

    def compare_all_of_task(cls, experiments_list):
        """
        Показує діалогове вікно порівняння експериментів, фільтруючи тільки ті,
        у яких test_metrics і train_metrics не рівні None.

        Args:
            experiments_list (list): Список експериментів для фільтрації та відображення

        Returns:
            None: Показує діалогове вікно або виводить повідомлення, якщо немає підходящих експериментів
        """
        # Фільтруємо експерименти, які мають як train_metrics, так і test_metrics
        filtered_experiments = []

        for experiment in experiments_list:
            # Перевіряємо чи існують атрибути train_metrics та test_metrics
            has_train_metrics = (hasattr(experiment, 'train_metrics') and
                                 experiment.train_metrics is not None)
            has_test_metrics = (hasattr(experiment, 'test_metrics') and
                                experiment.test_metrics is not None)

            # Додаємо експеримент тільки якщо обидві метрики присутні
            if has_train_metrics and has_test_metrics:
                filtered_experiments.append(experiment)

        # Перевіряємо чи є експерименти для відображення
        if not filtered_experiments:
            print("Немає експериментів з повними метриками (train_metrics та test_metrics) для відображення.")
            return

        print(f"Знайдено {len(filtered_experiments)} експериментів з повними метриками:")
        for exp in filtered_experiments:
            print(f"  - Експеримент ID: {exp.id}, Назва: {exp.name}, Задача: {exp.task}")



    def recreate_tables(self):
        """
        Пересоздає таблиці метрик з новими відфільтрованими експериментами.
        """
        # Очищаємо існуючі таблиці та пересоздаємо їх
        if hasattr(self, 'train_table'):
            # Отримуємо батьківський layout тренувальної таблиці
            train_parent_layout = self.train_table.parent().layout()
            if train_parent_layout:
                # Видаляємо стару таблицю
                train_parent_layout.removeWidget(self.train_table)
                self.train_table.deleteLater()

                # Створюємо нову таблицю
                self.train_table = self.create_metric_table("train_metrics")
                train_parent_layout.addWidget(self.train_table)

        if hasattr(self, 'test_table'):
            # Отримуємо батьківський layout тестової таблиці
            test_parent_layout = self.test_table.parent().layout()
            if test_parent_layout:
                # Видаляємо стару таблицю
                test_parent_layout.removeWidget(self.test_table)
                self.test_table.deleteLater()

                # Створюємо нову таблицю
                self.test_table = self.create_metric_table("test_metrics")
                test_parent_layout.addWidget(self.test_table)

    @staticmethod
    def create_dialog_with_filtered_experiments(experiments_list):
        """
        Статичний метод для створення нового діалогу з відфільтрованими експериментами.

        Args:
            experiments_list (list): Список експериментів для фільтрації

        Returns:
            ExperimentComparisonDialog або None: Новий діалог з відфільтрованими експериментами
                                                або None, якщо немає підходящих експериментів
        """
        # Фільтруємо експерименти
        filtered_experiments = []

        for experiment in experiments_list:
            has_train_metrics = (hasattr(experiment, 'train_metrics') and
                                 experiment.train_metrics is not None)
            has_test_metrics = (hasattr(experiment, 'test_metrics') and
                                experiment.test_metrics is not None)

            if has_train_metrics and has_test_metrics:
                filtered_experiments.append(experiment)

        # Перевіряємо чи є експерименти для відображення
        if not filtered_experiments:
            print("Немає експериментів з повними метриками для створення діалогу порівняння.")
            return None
        ExperimentComparisonDialog(filtered_experiments).exec_()
        # Створюємо новий діалог з відфільтрованими експериментами
        return