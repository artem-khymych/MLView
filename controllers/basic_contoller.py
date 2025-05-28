from PyQt5.QtWidgets import QApplication



class BasicController:
    def __init__(self, view):
        self.isMaximized = True
        self.view = view
        self.close_arrow = self.view.style().standardIcon(QApplication.style().SP_ArrowLeft)
        self.open_arrow = self.view.style().standardIcon(QApplication.style().SP_ArrowRight)

