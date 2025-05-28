from abc import abstractmethod
from PyQt5.QtCore import QObject
from PyQt5.QtWidgets import QWidget
from project.logic.experiment.experiment import Experiment



class TabController(QObject):
    def __init__(self, experiment: Experiment, view: QWidget):
        super().__init__()
        self.experiment = experiment
        self.view = view

    @abstractmethod
    def connect_signals(self):
        raise NotImplemented

    @abstractmethod
    def update_model_from_view(self):
        raise NotImplemented

    @abstractmethod
    def init_view(self):
        raise NotImplemented





