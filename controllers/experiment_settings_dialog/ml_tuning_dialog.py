import numpy as np
import ast
import inspect
import sys
from PyQt5.QtWidgets import (QApplication, QDialog, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QCheckBox, QPushButton,
                             QComboBox, QGroupBox, QScrollArea, QWidget,
                             QFormLayout, QSpinBox, QDoubleSpinBox, QGridLayout,
                             QTabWidget, QRadioButton, QButtonGroup, QMessageBox)
from PyQt5.QtCore import Qt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             silhouette_score, calinski_harabasz_score, davies_bouldin_score,
                             mean_squared_error, explained_variance_score, r2_score)
from sklearn.metrics import make_scorer
from sklearn.base import ClusterMixin, DensityMixin, OutlierMixin, is_regressor

class ParamTuningDialog(QDialog):
    def __init__(self, model, params, X_train=None, y_train=None, parent=None):
        super(ParamTuningDialog, self).__init__(parent)
        self.model = model
        self.params = params
        self.X_train = X_train
        self.y_train = y_train
        self.param_widgets = {}
        self.param_checkboxes = {}
        self.param_ranges = {}
        self.best_params = {}
        self.task_type = self._detect_task_type()

        self.initUI()

    def _detect_task_type(self):
        """Determine the task type based on model type"""
        if isinstance(self.model, ClusterMixin):
            return "clustering"
        elif any(isinstance(self.model, cls) for cls in [PCA, KernelPCA, TruncatedSVD]) or hasattr(self.model,
                                                                                                   'transform'):
            return "dimensionality_reduction"
        elif isinstance(self.model, DensityMixin):
            return "density_estimation"
        elif isinstance(self.model, OutlierMixin):
            return "anomaly_detection"
        elif is_regressor(self.model):
            return "regression"
        else:
            return "classification"

    def initUI(self):
        self.setWindowTitle('Налаштування параметрів моделі')  # "Model Parameters Tuning"
        self.setMinimumWidth(650)
        self.setMinimumHeight(1000)

        main_layout = QVBoxLayout()

        # Model information
        model_group = QGroupBox("Інформація про модель")  # "Model Information"
        model_layout = QFormLayout()
        model_name = QLabel(self.model.__class__.__name__)
        model_layout.addRow("Тип моделі:", model_name)  # "Model type:"

        task_type_label = QLabel(self._get_task_type_name())
        model_layout.addRow("Тип задачі:", task_type_label)  # "Task type:"

        model_group.setLayout(model_layout)
        main_layout.addWidget(model_group)

        # Search settings
        search_group = QGroupBox("Налаштування пошуку")  # "Search Settings"
        search_layout = QFormLayout()

        self.search_type = QComboBox()
        self.search_type.addItems(["GridSearchCV", "RandomizedSearchCV"])
        search_layout.addRow("Тип пошуку:", self.search_type)  # "Search type:"

        self.cv_spinbox = QSpinBox()
        self.cv_spinbox.setMinimum(2)
        self.cv_spinbox.setMaximum(20)
        self.cv_spinbox.setValue(5)
        search_layout.addRow("Кількість фолдів (CV):", self.cv_spinbox)  # "Number of folds (CV):"

        # Create metric selection widget based on task type
        self.metric_tab = QTabWidget()
        self._create_metric_tabs()
        search_layout.addRow("Метрика оцінки:", self.metric_tab)  # "Evaluation metric:"

        self.n_iter = QSpinBox()
        self.n_iter.setMinimum(10)
        self.n_iter.setMaximum(1000)
        self.n_iter.setValue(100)
        search_layout.addRow("Кількість ітерацій (для RandomizedSearchCV):", self.n_iter)  # "Number of iterations (for RandomizedSearchCV):"

        search_group.setLayout(search_layout)
        main_layout.addWidget(search_group)

        # Model parameters
        params_group = QGroupBox("Параметри моделі")  # "Model Parameters"
        params_layout = QVBoxLayout()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        for param_name, param_value in self.params.items():
            param_group = QGroupBox(param_name)
            param_layout = QVBoxLayout()

            # Checkbox for parameter selection
            checkbox = QCheckBox("Оптимізувати цей параметр")  # "Optimize this parameter"
            self.param_checkboxes[param_name] = checkbox
            param_layout.addWidget(checkbox)

            # Current value
            current_value = QLabel(f"Поточне значення: {param_value}")  # f"Current value: {param_value}"
            param_layout.addWidget(current_value)

            # Widgets for setting value ranges
            range_layout = QGridLayout()

            if isinstance(param_value, bool):
                # Boolean value
                self.param_ranges[param_name] = {"widget_type": "bool"}
                range_layout.addWidget(QLabel("Значення: [True, False]"), 0, 0, 1, 2)  # "Values: [True, False]"

            elif isinstance(param_value, int):
                # Integer parameter
                range_layout.addWidget(QLabel("Мін:"), 0, 0)  # "Min:"
                min_spin = QSpinBox()
                min_spin.setRange(-1000000, 1000000)
                min_spin.setValue(max(0, param_value - 5))
                range_layout.addWidget(min_spin, 0, 1)

                range_layout.addWidget(QLabel("Макс:"), 1, 0)  # "Max:"
                max_spin = QSpinBox()
                max_spin.setRange(-1000000, 1000000)
                max_spin.setValue(param_value + 5)
                range_layout.addWidget(max_spin, 1, 1)

                range_layout.addWidget(QLabel("Крок:"), 2, 0)  # "Step:"
                step_spin = QSpinBox()
                step_spin.setRange(1, 1000)
                step_spin.setValue(1)
                range_layout.addWidget(step_spin, 2, 1)

                self.param_ranges[param_name] = {
                    "widget_type": "int",
                    "min": min_spin,
                    "max": max_spin,
                    "step": step_spin
                }

            elif isinstance(param_value, float):
                # Float parameter
                range_layout.addWidget(QLabel("Мін:"), 0, 0)  # "Min:"
                min_spin = QDoubleSpinBox()
                min_spin.setRange(-1000000, 1000000)
                min_spin.setDecimals(6)
                min_spin.setValue(max(0, param_value - 0.5))
                range_layout.addWidget(min_spin, 0, 1)

                range_layout.addWidget(QLabel("Макс:"), 1, 0)  # "Max:"
                max_spin = QDoubleSpinBox()
                max_spin.setRange(-1000000, 1000000)
                max_spin.setDecimals(6)
                max_spin.setValue(param_value + 0.5)
                range_layout.addWidget(max_spin, 1, 1)

                range_layout.addWidget(QLabel("Крок:"), 2, 0)  # "Step:"
                step_spin = QDoubleSpinBox()
                step_spin.setRange(0.000001, 1000)
                step_spin.setDecimals(6)
                step_spin.setValue(0.1)
                range_layout.addWidget(step_spin, 2, 1)

                self.param_ranges[param_name] = {
                    "widget_type": "float",
                    "min": min_spin,
                    "max": max_spin,
                    "step": step_spin
                }

            elif isinstance(param_value, str):
                # String parameter
                range_layout.addWidget(QLabel("Варіанти (через кому):"), 0, 0)  # "Options (comma-separated):"
                options_edit = QLineEdit(param_value)
                range_layout.addWidget(options_edit, 0, 1)

                self.param_ranges[param_name] = {
                    "widget_type": "str",
                    "options": options_edit
                }

            else:
                # Other parameter types (lists, dicts etc.)
                range_layout.addWidget(QLabel("Варіанти (Python-синтаксис):"), 0, 0)  # "Options (Python syntax):"
                options_edit = QLineEdit(str(param_value))
                range_layout.addWidget(options_edit, 0, 1)

                self.param_ranges[param_name] = {
                    "widget_type": "complex",
                    "options": options_edit
                }

            param_layout.addLayout(range_layout)
            param_group.setLayout(param_layout)
            scroll_layout.addWidget(param_group)

        scroll.setWidget(scroll_widget)
        params_layout.addWidget(scroll)
        params_group.setLayout(params_layout)
        main_layout.addWidget(params_group)

        # Buttons
        buttons_layout = QHBoxLayout()
        self.run_button = QPushButton("Запустити пошук")  # "Run Search"
        self.run_button.clicked.connect(self.run_search)
        self.cancel_button = QPushButton("Скасувати")  # "Cancel"
        self.cancel_button.clicked.connect(self.reject)

        buttons_layout.addWidget(self.run_button)
        buttons_layout.addWidget(self.cancel_button)
        main_layout.addLayout(buttons_layout)

        self.setLayout(main_layout)

    def _get_task_type_name(self):
        """Get task type name for display"""
        task_type_names = {
            "classification": "Класифікація",  # "Classification"
            "Regression": "Регресія",  # "Regression"
            "clustering": "Кластеризація",  # "Clustering"
            "dimensionality_reduction": "Зменшення розмірності",  # "Dimensionality Reduction"
            "density_estimation": "Оцінка щільності",  # "Density Estimation"
            "anomaly_detection": "Виявлення аномалій"  # "Anomaly Detection"
        }
        return task_type_names.get(self.task_type, "Невідомий тип задачі")

    def _create_metric_tabs(self):
        """Створює вкладки з метриками для різних типів задач"""
        metrics = self._get_metrics_for_task()

        for category, category_metrics in metrics.items():
            tab = QWidget()
            tab_layout = QVBoxLayout(tab)

            button_group = QButtonGroup(tab)

            for i, (metric_name, metric_info) in enumerate(category_metrics.items()):
                radio = QRadioButton(f"{metric_name}: {metric_info['description']}")
                radio.setProperty("metric_name", metric_name)
                radio.setProperty("metric_func", metric_info.get("func"))
                radio.setProperty("greater_is_better", metric_info.get("greater_is_better", True))

                if i == 0:  # Перша метрика вибрана за замовчуванням
                    radio.setChecked(True)

                button_group.addButton(radio)
                tab_layout.addWidget(radio)

            # Додаємо еластичний простір внизу
            tab_layout.addStretch(1)

            self.metric_tab.addTab(tab, category)

    def _get_metrics_for_task(self):
        """Повертає словник метрик відповідно до типу задачі"""
        metrics = {}

        # Базові метрики для класифікації/регресії
        if self.task_type == "classification":
            metrics["Класифікація"] = {
                "accuracy": {"description": "Точність (частка правильних прогнозів)", "func": accuracy_score,
                             "greater_is_better": True},
                "precision": {"description": "Точність (precision)", "func": precision_score,
                              "greater_is_better": True},
                "recall": {"description": "Повнота (recall)", "func": recall_score, "greater_is_better": True},
                "f1": {"description": "F1 (середнє гармонійне precision і recall)", "func": f1_score,
                       "greater_is_better": True},
                "roc_auc": {"description": "Площа під ROC-кривою", "func": roc_auc_score, "greater_is_better": True}
            }
        elif self.task_type == "regression":
            metrics["Регресія"] = {
                "neg_mean_squared_error": {"description": "Негативна середня квадратична помилка",
                                           "func": lambda y, y_pred: -mean_squared_error(y, y_pred),
                                           "greater_is_better": True},
                "explained_variance": {"description": "Пояснена дисперсія", "func": explained_variance_score,
                                       "greater_is_better": True},
                "r2": {"description": "Коефіцієнт детермінації (R²)", "func": r2_score, "greater_is_better": True}
            }

        # Метрики для кластеризації
        elif self.task_type == "clustering":
            metrics["Кластеризація"] = {
                "silhouette": {
                    "description": "Силует (оцінює якість кластерів)",
                    "func": self._silhouette_scorer,
                    "greater_is_better": True
                },
                "calinski_harabasz": {
                    "description": "Індекс Calinski-Harabasz",
                    "func": self._calinski_harabasz_scorer,
                    "greater_is_better": True
                },
                "davies_bouldin": {
                    "description": "Індекс Davies-Bouldin",
                    "func": self._davies_bouldin_scorer,
                    "greater_is_better": False
                },
                "inertia": {
                    "description": "Інерція кластерів",
                    "func": self._inertia_scorer,
                    "greater_is_better": False
                }
            }

        # Метрики для зменшення розмірності
        elif self.task_type == "dimensionality_reduction":
            metrics["Якість проекції"] = {
                "explained_variance_ratio_sum": {
                    "description": "Сума пояснених дисперсій",
                    "func": self._explained_variance_ratio_scorer,
                    "greater_is_better": True
                },
                "reconstruction_error": {
                    "description": "Помилка реконструкції",
                    "func": self._reconstruction_error_scorer,
                    "greater_is_better": False
                }
            }

        # Метрики для оцінки щільності
        elif self.task_type == "density_estimation":
            metrics["Оцінка щільності"] = {
                "log_likelihood": {
                    "description": "Логарифм правдоподібності",
                    "func": self._log_likelihood_scorer,
                    "greater_is_better": True
                }
            }

        # Метрики для виявлення аномалій
        elif self.task_type == "anomaly_detection":
            metrics["Виявлення аномалій"] = {
                "outlier_score_mean": {
                    "description": "Середня оцінка аномальності (для порівняння моделей)",
                    "func": self._outlier_score_mean_scorer,
                    "greater_is_better": True
                },
                "outlier_score_std": {
                    "description": "Стандартне відхилення оцінок аномальності",
                    "func": self._outlier_score_std_scorer,
                    "greater_is_better": True
                }
            }

        # Якщо тип задачі невідомий, використовуємо стандартні метрики
        if not metrics:
            metrics["Стандартні"] = {
                "score": {
                    "description": "Стандартна метрика моделі",
                    "func": None,  # Буде використовуватись метод score() моделі
                    "greater_is_better": True
                }
            }

        return metrics

    # Функції-скорери для кластеризації
    def _silhouette_scorer(self, estimator, X, y=None):
        labels = estimator.predict(X)
        if len(np.unique(labels)) < 2:
            return -1  # Якщо тільки один кластер
        return silhouette_score(X, labels)

    def _calinski_harabasz_scorer(self, estimator, X, y=None):
        labels = estimator.predict(X)
        if len(np.unique(labels)) < 2:
            return 0  # Якщо тільки один кластер
        return calinski_harabasz_score(X, labels)

    def _davies_bouldin_scorer(self, estimator, X, y=None):
        labels = estimator.predict(X)
        if len(np.unique(labels)) < 2:
            return float('inf')  # Якщо тільки один кластер
        return -davies_bouldin_score(X, labels)  # Негативне значення, бо оптимізуємо на максимум

    def _inertia_scorer(self, estimator, X, y=None):
        estimator.fit(X)
        if hasattr(estimator, 'inertia_'):
            return -estimator.inertia_  # Негативне значення, бо оптимізуємо на максимум
        return 0

    def _cluster_size_std_scorer(self, estimator, X, y=None):
        labels = estimator.predict(X)
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return float('inf')

        cluster_sizes = [np.sum(labels == label) for label in unique_labels]
        return -np.std(cluster_sizes)  # Негативне значення, бо оптимізуємо на максимум

    def _min_cluster_size_scorer(self, estimator, X, y=None):
        labels = estimator.predict(X)
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return 0

        cluster_sizes = [np.sum(labels == label) for label in unique_labels]
        return min(cluster_sizes)

    # Функції-скорери для зменшення розмірності
    def _explained_variance_ratio_scorer(self, estimator, X, y=None):
        estimator.fit(X)
        if hasattr(estimator, 'explained_variance_ratio_'):
            return np.sum(estimator.explained_variance_ratio_)
        return 0

    def _reconstruction_error_scorer(self, estimator, X, y=None):
        estimator.fit(X)
        if hasattr(estimator, 'transform') and hasattr(estimator, 'inverse_transform'):
            X_reduced = estimator.transform(X)
            X_reconstructed = estimator.inverse_transform(X_reduced)
            return -np.mean(np.square(X - X_reconstructed))  # Негативне значення, бо оптимізуємо на максимум
        return -float('inf')

    # Функції-скорери для оцінки щільності
    def _log_likelihood_scorer(self, estimator, X, y=None):
        estimator.fit(X)
        if hasattr(estimator, 'score'):
            return estimator.score(X)
        return 0

    # Функції-скорери для виявлення аномалій
    def _outlier_score_mean_scorer(self, estimator, X, y=None):
        estimator.fit(X)
        if hasattr(estimator, 'decision_function'):
            scores = estimator.decision_function(X)
            return np.mean(scores)
        elif hasattr(estimator, 'score_samples'):
            scores = estimator.score_samples(X)
            return np.mean(scores)
        return 0

    def _outlier_score_std_scorer(self, estimator, X, y=None):
        estimator.fit(X)
        if hasattr(estimator, 'decision_function'):
            scores = estimator.decision_function(X)
            return np.std(scores)
        elif hasattr(estimator, 'score_samples'):
            scores = estimator.score_samples(X)
            return np.std(scores)
        return 0

    def get_selected_metric(self):
        """Отримати вибрану метрику"""
        current_tab = self.metric_tab.currentWidget()

        for radio in current_tab.findChildren(QRadioButton):
            if radio.isChecked():
                metric_name = radio.property("metric_name")
                metric_func = radio.property("metric_func")
                greater_is_better = radio.property("greater_is_better")

                return {
                    "name": metric_name,
                    "func": metric_func,
                    "greater_is_better": greater_is_better
                }

        return {"name": "score", "func": None, "greater_is_better": True}

    def get_param_grid(self):
        param_grid = {}

        for param_name, checkbox in self.param_checkboxes.items():
            if checkbox.isChecked():
                param_config = self.param_ranges[param_name]

                if param_config["widget_type"] == "bool":
                    param_grid[param_name] = [True, False]

                elif param_config["widget_type"] == "int":
                    min_val = param_config["min"].value()
                    max_val = param_config["max"].value()
                    step = param_config["step"].value()
                    param_grid[param_name] = list(range(min_val, max_val + 1, step))

                elif param_config["widget_type"] == "float":
                    min_val = param_config["min"].value()
                    max_val = param_config["max"].value()
                    step = param_config["step"].value()
                    values = []
                    current = min_val
                    while current <= max_val:
                        values.append(current)
                        current += step
                    param_grid[param_name] = values

                elif param_config["widget_type"] in ["str", "complex"]:
                    try:
                        options_text = param_config["options"].text()
                        if "," in options_text:
                            # Список через кому
                            options = [opt.strip() for opt in options_text.split(",")]
                        else:
                            # Python-синтаксис
                            options = ast.literal_eval(options_text)
                            if not isinstance(options, list):
                                options = [options]
                        param_grid[param_name] = options
                    except Exception as e:
                        print(f"Помилка обробки параметра {param_name}: {e}")
                        continue

        return param_grid

    def run_search(self):
        param_grid = self.get_param_grid()

        if not param_grid:
            QMessageBox.warning(self, "Попередження",
                                "Виберіть хоча б один параметр для оптимізації!")
            return

        if self.X_train is None:
            QMessageBox.critical(self, "Помилка",
                                 "Відсутні дані для тренування (X_train)!")
            return

        try:
            cv = self.cv_spinbox.value()
            selected_metric = self.get_selected_metric()

            scoring = None
            if selected_metric["func"] is not None:
                scoring = make_scorer(
                    selected_metric["func"],
                    greater_is_better=selected_metric["greater_is_better"]
                )

            if self.search_type.currentText() == "GridSearchCV":
                search = GridSearchCV(
                    self.model,
                    param_grid,
                    cv=cv if self.task_type == "classification_regression" and self.y_train is not None else None,
                    scoring=scoring,
                    n_jobs=1,
                    verbose=1
                )
            else:  # RandomizedSearchCV
                n_iter = self.n_iter.value()
                search = RandomizedSearchCV(
                    self.model,
                    param_grid,
                    n_iter=n_iter,
                    cv=cv if self.task_type == "classification_regression" and self.y_train is not None else None,
                    scoring=scoring,
                    n_jobs=-1,
                    verbose=1
                )

            progress = QMessageBox()
            progress.setWindowTitle("Виконується пошук")
            progress.setText("Виконується пошук оптимальних параметрів...\nЦе може зайняти деякий час.")
            progress.setStandardButtons(QMessageBox.NoButton)
            progress.show()
            QApplication.processEvents()

            # Для кластеризації та інших задач без y_train
            if self.task_type == "classification" or self.task_type == "regression":
                search.fit(self.X_train, self.y_train)
            else:
                search.fit(self.X_train)

            progress.close()

            self.best_params = search.best_params_
            best_score = search.best_score_

            result_message = f"Найкращий результат ({selected_metric['name']}): {best_score:.4f}\n\n"
            result_message += "Найкращі параметри:\n"
            for param, value in self.best_params.items():
                result_message += f"{param}: {value}\n"

            QMessageBox.information(self, "Результати пошуку", result_message)
            self.accept()

        except Exception as e:
            QMessageBox.critical(self, "Помилка", f"Виникла помилка під час пошуку: {str(e)}")

    def get_best_params(self):
        return self.best_params


def show_param_tuning_dialog(model, params, X_train=None, y_train=None):
    """
    Shows dialog for tuning machine learning model parameters

    Args:
        model: sklearn model to optimize
        params: Dictionary with current model parameters
        X_train: Training data
        y_train: Target variable for training (optional for clustering and some other tasks)

    Returns:
        dict: Dictionary with optimal parameters or empty dict if user canceled dialog
    """
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    dialog = ParamTuningDialog(model, params, X_train, y_train)
    result = dialog.exec_()

    if result == QDialog.Accepted:
        return dialog.get_best_params()
    else:
        return {}
