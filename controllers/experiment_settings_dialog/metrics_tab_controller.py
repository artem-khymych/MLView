from PyQt5.QtCore import pyqtSignal

from project.controllers.experiment_settings_dialog.tab_controller import TabController
from project.logic.experiment.experiment import Experiment


class MetricsTabController(TabController):
    """Controller for evaluation metrics tab"""
    compare_experiments = pyqtSignal(int)
    compare_all = pyqtSignal(object)

    def __init__(self, experiment: Experiment, view):
        super().__init__(experiment, view)
        self.experiment = experiment
        self.connect_signals()
        self.init_view()

    def connect_signals(self):
        """Connect signals to slots"""
        # Connect experiment signals if available
        if hasattr(self.experiment, 'metrics_updated'):
            self.experiment.metrics_updated.connect(self.on_metrics_updated)
        self.view.compare_button.clicked.connect(self.on_compare_button_clicked)
        self.view.compare_all.clicked.connect(self.on_compare_all_clicked)

    def init_view(self):
        """Initialize the view"""
        # Get current metrics from experiment if they exist
        train_metrics = getattr(self.experiment, 'train_metrics', {})
        test_metrics = getattr(self.experiment, 'test_metrics', {})

        if train_metrics or test_metrics:
            self.update_view_metrics(train_metrics, test_metrics)

    def on_metrics_updated(self, train_metrics=None, test_metrics=None):
        """Handler for metrics update event"""
        self.update_view_metrics(train_metrics, test_metrics)

    def update_view_metrics(self, train_metrics=None, test_metrics=None):
        """Update metrics display in the view"""
        if train_metrics is None:
            train_metrics = {}
        if test_metrics is None:
            test_metrics = {}

        # Combine unique keys from both dictionaries
        all_metrics = set(list(train_metrics.keys()) + list(test_metrics.keys()))
        metrics_data = {}

        for metric in all_metrics:
            metrics_data[metric] = {
                'train': train_metrics.get(metric, None),
                'test': test_metrics.get(metric, None)
            }

        self.view.update_metrics(metrics_data)

    def on_compare_button_clicked(self):
        """Handler for compare button click event"""
        self.compare_experiments.emit(self.experiment.id)

    def on_compare_all_clicked(self):
        self.compare_all.emit(self.experiment.task)