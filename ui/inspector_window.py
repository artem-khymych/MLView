from PyQt5.QtWidgets import (QListWidget
)


from project.ui.basic_window import BasicWindow


class InspectorWindow(BasicWindow):
    """Class for representing an inspector of experiments"""
    def __init__(self):
        super().__init__()

        self.object_list = QListWidget()
        self.object_list.addItems(["Об'єкт 1", "Об'єкт 2", "Об'єкт 3", "Об'єкт 4"])
        self.layout.addWidget(self.object_list)
        self.layout.addStretch()


