import sys

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication
from project.controllers.main_controller import MainController
from project.ui.styles.styles import setup_app_style


def main():
    app = QApplication(sys.argv)
    controller = MainController()
    setup_app_style(app)
    icon = QIcon("resources/icon.ico")
    app.setWindowIcon(icon)
    controller.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
