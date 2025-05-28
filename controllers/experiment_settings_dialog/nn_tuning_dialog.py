
import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QLineEdit, QComboBox, QPushButton, QGroupBox, QCheckBox,
                             QDoubleSpinBox, QSpinBox, QTabWidget, QProgressBar, QTextEdit,
                             QScrollArea, QWidget, QGridLayout, QSizePolicy, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
import tensorflow as tf
from itertools import product
import time


class OptimizationWorker(QThread):
    update_progress = pyqtSignal(int)
    update_results = pyqtSignal(str)
    optimization_complete = pyqtSignal(dict, dict)

    def __init__(self, model, params_to_tune, X_train, y_train):
        super().__init__()
        self.model = model
        self.params_to_tune = params_to_tune
        self.X_train = X_train
        self.y_train = y_train
        self.best_compile_params = {}
        self.best_fit_params = {}
        self.best_score = float('-inf')

    def run(self):
        # Generate all combinations of parameters
        compile_params_keys = list(self.params_to_tune['compile'].keys())
        compile_params_values = list(self.params_to_tune['compile'].values())

        fit_params_keys = list(self.params_to_tune['fit'].keys())
        fit_params_values = list(self.params_to_tune['fit'].values())

        compile_combinations = list(product(*compile_params_values))
        fit_combinations = list(product(*fit_params_values))

        total_combinations = len(compile_combinations) * len(fit_combinations)
        processed = 0

        for compile_combo in compile_combinations:
            compile_params = {compile_params_keys[i]: compile_combo[i] for i in range(len(compile_params_keys))}

            for fit_combo in fit_combinations:
                fit_params = {fit_params_keys[i]: fit_combo[i] for i in range(len(fit_params_keys))}

                # Clone the model to avoid retraining the same model
                model_clone = tf.keras.models.clone_model(self.model)

                # Compile the model with current parameters
                if 'optimizer' in compile_params:
                    optimizer_name = compile_params['optimizer']
                    if optimizer_name == 'adam':
                        optimizer = tf.keras.optimizers.Adam(
                            learning_rate=compile_params.get('learning_rate', 0.001)
                        )
                    elif optimizer_name == 'sgd':
                        optimizer = tf.keras.optimizers.SGD(
                            learning_rate=compile_params.get('learning_rate', 0.01)
                        )
                    elif optimizer_name == 'rmsprop':
                        optimizer = tf.keras.optimizers.RMSprop(
                            learning_rate=compile_params.get('learning_rate', 0.001)
                        )
                    elif optimizer_name == 'adagrad':
                        optimizer = tf.keras.optimizers.    Adagrad(
                            learning_rate=compile_params.get('learning_rate', 0.001)
                        )
                    elif optimizer_name == 'adadelta':
                        optimizer = tf.keras.optimizers.Adadelta(
                            learning_rate=compile_params.get('learning_rate', 0.001)
                        )
                    elif optimizer_name == 'adamax':
                        optimizer = tf.keras.optimizers.Adamax(
                            learning_rate=compile_params.get('learning_rate', 0.001)
                        )
                    elif optimizer_name == 'nadam':
                        optimizer = tf.keras.optimizers.Nadam(
                            learning_rate=compile_params.get('learning_rate', 0.001)
                        )
                    compile_params_copy = compile_params.copy()
                    compile_params_copy['optimizer'] = optimizer
                else:
                    compile_params_copy = compile_params.copy()

                model_clone.compile(**compile_params_copy)

                # Prepare validation data if needed
                validation_data = None
                if fit_params.get('validation_split', 0) > 0:
                    # Remove validation_split from fit_params as we'll use validation_data instead
                    val_split = fit_params.pop('validation_split', 0.2)
                    # Calculate split index
                    split_idx = int(len(self.X_train) * (1 - val_split))
                    validation_data = (self.X_train[split_idx:], self.y_train[split_idx:])
                    X_train_part = self.X_train[:split_idx]
                    y_train_part = self.y_train[:split_idx]
                else:
                    X_train_part = self.X_train
                    y_train_part = self.y_train

                # Train the model
                try:
                    history = model_clone.fit(
                        X_train_part,
                        y_train_part,
                        validation_data=validation_data,
                        **fit_params
                    )

                    # Get the best validation accuracy (or other metric)
                    if validation_data is not None:
                        if 'val_accuracy' in history.history:
                            score = max(history.history['val_accuracy'])
                        elif 'val_acc' in history.history:
                            score = max(history.history['val_acc'])
                        else:
                            # If no validation accuracy, use training accuracy
                            score = max(
                                history.history['accuracy'] if 'accuracy' in history.history else history.history[
                                    'acc'])
                    else:
                        # Use training accuracy if no validation
                        score = max(
                            history.history['accuracy'] if 'accuracy' in history.history else history.history['acc'])

                    # Update best params if current model is better
                    if score > self.best_score:
                        self.best_score = score
                        self.best_compile_params = compile_params
                        self.best_fit_params = fit_params

                        # Emit update to log
                        self.update_results.emit(
                            f"Нова найкраща модель: точність = {score:.4f}\n"
                            f"Compile параметри: {compile_params}\n"
                            f"Fit параметри: {fit_params}\n"
                            f"{'-' * 50}\n"
                        )

                except Exception as e:
                    self.update_results.emit(f"Помилка під час навчання: {str(e)}\n")

                # Update progress
                processed += 1
                progress = int((processed / total_combinations) * 100)
                self.update_progress.emit(progress)

        # Emit the best parameters found
        self.optimization_complete.emit(self.best_compile_params, self.best_fit_params)


class ModelOptimizerDialog(QDialog):
    def __init__(self, model, X_train, y_train, task, parent=None):
        super().__init__(parent)
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.params_to_tune = {
            'compile': {},
            'fit': {}
        }
        self.task = task

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Оптимізація параметрів нейромережевої моделі')
        self.setMinimumSize(700, 600)

        main_layout = QVBoxLayout()

        self.tabs = QTabWidget()
        self.compile_tab = QWidget()
        self.fit_tab = QWidget()
        self.results_tab = QWidget()

        self.setup_compile_tab()
        self.setup_fit_tab()
        self.setup_results_tab()

        self.tabs.addTab(self.compile_tab, "Compile параметри")
        self.tabs.addTab(self.fit_tab, "Fit параметри")
        self.tabs.addTab(self.results_tab, "Результати")

        main_layout.addWidget(self.tabs)

        buttons_layout = QHBoxLayout()

        self.start_button = QPushButton("Почати оптимізацію")
        self.start_button.clicked.connect(self.start_optimization)

        self.accept_button = QPushButton("Закрити і зберегти")
        self.accept_button.clicked.connect(self.accept)

        buttons_layout.addWidget(self.start_button)
        buttons_layout.addWidget(self.accept_button)

        main_layout.addLayout(buttons_layout)

        self.setLayout(main_layout)

    def setup_compile_tab(self):
        layout = QVBoxLayout()

        # Scroll area для великої кількості параметрів
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content_widget = QWidget()
        scroll_layout = QVBoxLayout(content_widget)

        # Оптимізатор
        optimizer_group = QGroupBox("Оптимізатор")
        optimizer_layout = QVBoxLayout()


        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["adam", "sgd", "rmsprop", "adagrad", "adadelta", "adamax", "nadam"])

        optimizer_layout.addWidget(self.optimizer_combo)
        optimizer_group.setLayout(optimizer_layout)
        scroll_layout.addWidget(optimizer_group)

        # Learning rate
        lr_group = QGroupBox("Learning Rate")
        lr_layout = QGridLayout()

        self.use_lr = QCheckBox("Налаштувати Learning Rate")
        self.use_lr.toggled.connect(self.toggle_lr_inputs)

        self.lr_min = QDoubleSpinBox()
        self.lr_min.setRange(0.0001, 1.0)
        self.lr_min.setSingleStep(0.0001)
        self.lr_min.setValue(0.0001)
        self.lr_min.setDecimals(6)

        self.lr_max = QDoubleSpinBox()
        self.lr_max.setRange(0.0001, 1.0)
        self.lr_max.setSingleStep(0.0001)
        self.lr_max.setValue(0.01)
        self.lr_max.setDecimals(6)

        self.lr_step = QDoubleSpinBox()
        self.lr_step.setRange(0.0001, 0.1)
        self.lr_step.setSingleStep(0.0001)
        self.lr_step.setValue(0.001)
        self.lr_step.setDecimals(6)

        lr_layout.addWidget(self.use_lr, 0, 0, 1, 2)
        lr_layout.addWidget(QLabel("Мінімум:"), 1, 0)
        lr_layout.addWidget(self.lr_min, 1, 1)
        lr_layout.addWidget(QLabel("Максимум:"), 2, 0)
        lr_layout.addWidget(self.lr_max, 2, 1)
        lr_layout.addWidget(QLabel("Крок:"), 3, 0)
        lr_layout.addWidget(self.lr_step, 3, 1)

        lr_group.setLayout(lr_layout)
        scroll_layout.addWidget(lr_group)

        loss_group = QGroupBox("Функція втрат (Loss)")
        loss_layout = QVBoxLayout()

        self.loss_combo = QComboBox()
        self.loss_combo.addItems(["binary_crossentropy", "sparse_categorical_crossentropy", "categorical_crossentropy", "mse", "mae", "mape", "cosine_similarity"])

        loss_layout.addWidget(self.loss_combo)
        loss_group.setLayout(loss_layout)
        scroll_layout.addWidget(loss_group)

        # Метрики
        metrics_group = QGroupBox("Метрики")
        metrics_layout = QVBoxLayout()

        self.metrics_accuracy = QCheckBox("accuracy")
        self.metrics_precision = QCheckBox("precision")
        self.metrics_recall = QCheckBox("recall")
        self.metrics_auc = QCheckBox("AUC")
        self.metrics_mse = QCheckBox("mse")
        self.metrics_mae = QCheckBox("mae")

        metrics_layout.addWidget(self.metrics_accuracy)
        metrics_layout.addWidget(self.metrics_precision)
        metrics_layout.addWidget(self.metrics_recall)
        metrics_layout.addWidget(self.metrics_auc)

        metrics_group.setLayout(metrics_layout)
        scroll_layout.addWidget(metrics_group)

        scroll.setWidget(content_widget)
        layout.addWidget(scroll)
        self.compile_tab.setLayout(layout)

        # Початкові стани
        self.toggle_lr_inputs(False)

    def setup_fit_tab(self):
        layout = QVBoxLayout()

        # Scroll area для великої кількості параметрів
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content_widget = QWidget()
        scroll_layout = QVBoxLayout(content_widget)

        # Batch size
        batch_group = QGroupBox("Batch Size")
        batch_layout = QGridLayout()

        self.use_batch = QCheckBox("Налаштувати Batch Size")
        self.use_batch.toggled.connect(self.toggle_batch_inputs)

        self.batch_min = QSpinBox()
        self.batch_min.setRange(1, 1024)
        self.batch_min.setValue(16)

        self.batch_max = QSpinBox()
        self.batch_max.setRange(1, 1024)
        self.batch_max.setValue(128)

        self.batch_step = QSpinBox()
        self.batch_step.setRange(1, 128)
        self.batch_step.setValue(16)

        batch_layout.addWidget(self.use_batch, 0, 0, 1, 2)
        batch_layout.addWidget(QLabel("Мінімум:"), 1, 0)
        batch_layout.addWidget(self.batch_min, 1, 1)
        batch_layout.addWidget(QLabel("Максимум:"), 2, 0)
        batch_layout.addWidget(self.batch_max, 2, 1)
        batch_layout.addWidget(QLabel("Крок:"), 3, 0)
        batch_layout.addWidget(self.batch_step, 3, 1)

        batch_group.setLayout(batch_layout)
        scroll_layout.addWidget(batch_group)

        # Epochs
        epochs_group = QGroupBox("Кількість епох")
        epochs_layout = QGridLayout()

        self.use_epochs = QCheckBox("Налаштувати кількість епох")
        self.use_epochs.toggled.connect(self.toggle_epochs_inputs)

        self.epochs_min = QSpinBox()
        self.epochs_min.setRange(1, 1000)
        self.epochs_min.setValue(5)

        self.epochs_max = QSpinBox()
        self.epochs_max.setRange(1, 1000)
        self.epochs_max.setValue(50)

        self.epochs_step = QSpinBox()
        self.epochs_step.setRange(1, 50)
        self.epochs_step.setValue(5)

        epochs_layout.addWidget(self.use_epochs, 0, 0, 1, 2)
        epochs_layout.addWidget(QLabel("Мінімум:"), 1, 0)
        epochs_layout.addWidget(self.epochs_min, 1, 1)
        epochs_layout.addWidget(QLabel("Максимум:"), 2, 0)
        epochs_layout.addWidget(self.epochs_max, 2, 1)
        epochs_layout.addWidget(QLabel("Крок:"), 3, 0)
        epochs_layout.addWidget(self.epochs_step, 3, 1)

        epochs_group.setLayout(epochs_layout)
        scroll_layout.addWidget(epochs_group)

        # Validation Split
        val_group = QGroupBox("Validation Split")
        val_layout = QGridLayout()

        self.use_val_split = QCheckBox("Налаштувати Validation Split")
        self.use_val_split.toggled.connect(self.toggle_val_split_inputs)

        self.val_split_min = QDoubleSpinBox()
        self.val_split_min.setRange(0.1, 0.5)
        self.val_split_min.setSingleStep(0.05)
        self.val_split_min.setValue(0.1)
        self.val_split_min.setDecimals(2)

        self.val_split_max = QDoubleSpinBox()
        self.val_split_max.setRange(0.1, 0.5)
        self.val_split_max.setSingleStep(0.05)
        self.val_split_max.setValue(0.3)
        self.val_split_max.setDecimals(2)

        self.val_split_step = QDoubleSpinBox()
        self.val_split_step.setRange(0.05, 0.2)
        self.val_split_step.setSingleStep(0.05)
        self.val_split_step.setValue(0.1)
        self.val_split_step.setDecimals(2)

        val_layout.addWidget(self.use_val_split, 0, 0, 1, 2)
        val_layout.addWidget(QLabel("Мінімум:"), 1, 0)
        val_layout.addWidget(self.val_split_min, 1, 1)
        val_layout.addWidget(QLabel("Максимум:"), 2, 0)
        val_layout.addWidget(self.val_split_max, 2, 1)
        val_layout.addWidget(QLabel("Крок:"), 3, 0)
        val_layout.addWidget(self.val_split_step, 3, 1)

        val_group.setLayout(val_layout)
        scroll_layout.addWidget(val_group)

        # Callbacks - Early Stopping
        callbacks_group = QGroupBox("Early Stopping")
        callbacks_layout = QVBoxLayout()

        self.use_early_stopping = QCheckBox("Використовувати Early Stopping")
        self.use_early_stopping.toggled.connect(lambda checked: self.early_stopping_patience.setEnabled(checked))

        patience_layout = QHBoxLayout()
        patience_layout.addWidget(QLabel("Patience:"))
        self.early_stopping_patience = QSpinBox()
        self.early_stopping_patience.setRange(1, 50)
        self.early_stopping_patience.setValue(10)
        self.early_stopping_patience.setEnabled(False)
        patience_layout.addWidget(self.early_stopping_patience)

        callbacks_layout.addWidget(self.use_early_stopping)
        callbacks_layout.addLayout(patience_layout)
        callbacks_group.setLayout(callbacks_layout)
        scroll_layout.addWidget(callbacks_group)

        # Завершення налаштування вкладки
        scroll.setWidget(content_widget)
        layout.addWidget(scroll)
        self.fit_tab.setLayout(layout)

        # Початкові стани
        self.toggle_batch_inputs(False)
        self.toggle_epochs_inputs(False)
        self.toggle_val_split_inputs(False)

    def setup_results_tab(self):
        layout = QVBoxLayout()

        self.progress_label = QLabel("Прогрес підбору параметрів:")
        layout.addWidget(self.progress_label)

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # Поле для логу результатів
        self.log_label = QLabel("Журнал підбору параметрів:")
        layout.addWidget(self.log_label)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

        self.best_params_label = QLabel("Найкращі параметри:")
        layout.addWidget(self.best_params_label)

        self.best_params_text = QTextEdit()
        self.best_params_text.setReadOnly(True)
        self.best_params_text.setMaximumHeight(150)
        layout.addWidget(self.best_params_text)

        self.results_tab.setLayout(layout)

    def toggle_lr_inputs(self, enabled):
        self.lr_min.setEnabled(enabled)
        self.lr_max.setEnabled(enabled)
        self.lr_step.setEnabled(enabled)

    def toggle_metrics(self, enabled):
        self.metrics_accuracy.setEnabled(enabled)
        self.metrics_precision.setEnabled(enabled)
        self.metrics_recall.setEnabled(enabled)
        self.metrics_auc.setEnabled(enabled)

    def toggle_batch_inputs(self, enabled):
        self.batch_min.setEnabled(enabled)
        self.batch_max.setEnabled(enabled)
        self.batch_step.setEnabled(enabled)

    def toggle_epochs_inputs(self, enabled):
        self.epochs_min.setEnabled(enabled)
        self.epochs_max.setEnabled(enabled)
        self.epochs_step.setEnabled(enabled)

    def toggle_val_split_inputs(self, enabled):
        self.val_split_min.setEnabled(enabled)
        self.val_split_max.setEnabled(enabled)
        self.val_split_step.setEnabled(enabled)

    def collect_parameters(self):
        # Збираємо параметри для compile
        compile_params = {}


        compile_params['optimizer'] = [self.optimizer_combo.currentText()]

        if self.use_lr.isChecked():
            min_lr = self.lr_min.value()
            max_lr = self.lr_max.value()
            step_lr = self.lr_step.value()
            lr_values = np.arange(min_lr, max_lr + step_lr, step_lr).tolist()
            compile_params['learning_rate'] = lr_values


        compile_params['loss'] = [self.loss_combo.currentText()]

        metrics = []
        if self.metrics_accuracy.isChecked():
            metrics.append('accuracy')
        if self.metrics_precision.isChecked():
            metrics.append('precision')
        if self.metrics_recall.isChecked():
            metrics.append('recall')
        if self.metrics_auc.isChecked():
            metrics.append('AUC')
        if self.metrics_mae.isChecked():
            metrics.append('mae')
        if self.metrics_mse.isChecked():
            metrics.append('mse')

        if not metrics:
            QMessageBox.warning(self, "Помилка", "Необхідно обрати хоча б одну метрику!")
            return None

        if metrics:
            compile_params['metrics'] = [metrics]  # Список метрик

        fit_params = {}

        if self.use_batch.isChecked():
            min_batch = self.batch_min.value()
            max_batch = self.batch_max.value()
            step_batch = self.batch_step.value()
            batch_values = list(range(min_batch, max_batch + step_batch, step_batch))
            fit_params['batch_size'] = batch_values

        if self.use_epochs.isChecked():
            min_epochs = self.epochs_min.value()
            max_epochs = self.epochs_max.value()
            step_epochs = self.epochs_step.value()
            epochs_values = list(range(min_epochs, max_epochs + step_epochs, step_epochs))
            fit_params['epochs'] = epochs_values

        if self.use_val_split.isChecked():
            min_val = self.val_split_min.value()
            max_val = self.val_split_max.value()
            step_val = self.val_split_step.value()
            val_values = np.arange(min_val, max_val + step_val, step_val).tolist()
            fit_params['validation_split'] = val_values

        if self.use_early_stopping.isChecked():
            # Early stopping буде додано окремо під час навчання
            patience = self.early_stopping_patience.value()
            fit_params['callbacks'] = [[tf.keras.callbacks.EarlyStopping(
                monitor='val_loss' if 'validation_split' in fit_params else 'loss',
                patience=patience,
                restore_best_weights=True
            )]]

        if not compile_params:
            compile_params = {'optimizer': ['adam'], 'loss': ['categorical_crossentropy'], 'metrics': [['accuracy']]}

        if not fit_params:
            fit_params = {'batch_size': [32], 'epochs': [10]}

        return {'compile': compile_params, 'fit': fit_params}

    def start_optimization(self):
        self.params_to_tune = self.collect_parameters()

        if self.params_to_tune is None:
            return
        self.tabs.setCurrentIndex(2)

        # Скидаємо прогрес-бар
        self.progress_bar.setValue(0)

        # Очищаємо лог
        self.log_text.clear()
        self.best_params_text.clear()

        self.worker = OptimizationWorker(self.model, self.params_to_tune, self.X_train, self.y_train)
        self.worker.update_progress.connect(self.update_progress)
        self.worker.update_results.connect(self.update_results)
        self.worker.optimization_complete.connect(self.optimization_complete)

        self.start_button.setEnabled(False)
        self.start_button.setText("Оптимізація в процесі...")

        self.log_text.append("Починаємо підбір оптимальних параметрів...\n")

        self.worker.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def update_results(self, text):
        self.log_text.append(text)
        sb = self.log_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def optimization_complete(self, best_compile_params, best_fit_params):
        self.log_text.append("Оптимізацію завершено!\n")

        self.start_button.setEnabled(True)
        self.start_button.setText("Почати оптимізацію")

        best_params_text = "НАЙКРАЩІ ПАРАМЕТРИ:\n\n"
        best_params_text += "Compile параметри:\n"
        for key, value in best_compile_params.items():
            best_params_text += f"{key}: {value}\n"

        best_params_text += "\nFit параметри:\n"
        for key, value in best_fit_params.items():
            if key == 'callbacks':
                best_params_text += f"{key}: [EarlyStopping(patience={self.early_stopping_patience.value()})]\n"
            else:
                best_params_text += f"{key}: {value}\n"

        self.best_params_text.setText(best_params_text)

        self.best_compile_params = best_compile_params
        self.best_fit_params = best_fit_params


def show_nn_tuning_dialog(model, X_train=None, y_train=None, task = None):
    """
    Показує діалог для налаштування оптимізації параметрів моделі.

    Parameters:
    -----------
    model : tf.keras.Model
        Модель TensorFlow для оптимізації.
    X_train : numpy.ndarray
        Навчальні дані.
    y_train : numpy.ndarray
        Цільові значення.

    Returns:
    --------
    tuple
        (compile_params, fit_params) - словники з оптимальними параметрами
        або (None, None), якщо користувач закрив діалог.
    """
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    dialog = ModelOptimizerDialog(model, X_train, y_train, task)
    result = dialog.exec_()

    if result == QDialog.Accepted:
        return dialog.best_compile_params, dialog.best_fit_params
    else:
        return None, None
