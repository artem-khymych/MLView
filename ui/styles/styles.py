from PyQt5.QtCore import QSize, QObject
from PyQt5.QtGui import QFont, QColor, QPalette
from PyQt5.QtWidgets import QLabel, QMainWindow


def setup_app_style(app):

    stylesheet = """
    /* Загальні стилі для всіх віджетів */
    QWidget {
        background-color: #F5F7FA;
        color: #2C3E50;
        font-family: 'Segoe UI', 'Arial', sans-serif;
        font-size: 10pt;
    }

    /* Заголовки та мітки */
    QLabel {
        color: #34495E;
        padding: 2px;
    }

    QLabel[heading="true"] {
        font-size: 14pt;
        font-weight: bold;
        color: #2C3E50;
        padding-bottom: 8px;
    }

    /* Поля введення */
    QLineEdit, QTextEdit, QPlainTextEdit {
        border: 1px solid #BDC3C7;
        border-radius: 3px;
        padding: 5px;
        background-color: white;
        selection-background-color: #3498DB;
        selection-color: white;
    }

    QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {
        border: 1px solid #3498DB;
    }

    /* Кнопки */
    QPushButton {
        background-color: #3498DB;
        color: white;
        border-radius: 3px;
        padding: 6px 12px;
        font-weight: bold;
        min-width: 80px;
    }

    QPushButton:hover {
        background-color: #2980B9;
    }

    QPushButton:pressed {
        background-color: #1F618D;
    }
    /* Додайте це до вашого stylesheet */

    QPushButton:disabled {
        background-color: #BDC3C7;
        color: #7F8C8D;
    }

    /* Кнопка для створення нового експерименту з особливим стилем */
    QPushButton#new_experiment_button {
        background-color: #2ECC71;
        font-weight: bold;
    }

    QPushButton#new_experiment_button:hover {
        background-color: #27AE60;
    }

    QPushButton#new_experiment_button:pressed {
        background-color: #1E8449;
    }

    /* Кнопка для запуску експерименту */
    QPushButton#start_button {
        background-color: #2ECC71;
    }

    QPushButton#start_button:hover {
        background-color: #27AE60;
    }

    QPushButton#start_button:pressed {
        background-color: #1E8449;
    }

    /* Кнопка оцінки */
    QPushButton#evaluate_button {
        background-color: #3498DB;
    }

    /* Кнопка успадкування */
    QPushButton#inherit_button {
        background-color: #9B59B6;
    }

    QPushButton#inherit_button:hover {
        background-color: #8E44AD;
    }

    /* Кнопка збереження */
    QPushButton#save_button {
        background-color: #F39C12;
    }

    QPushButton#save_button:hover {
        background-color: #E67E22;
    }

    /* Списки */
    QListWidget, QTreeWidget, QListView, QTreeView {
        background-color: white;
        alternate-background-color: #EFF0F1;
        border: 1px solid #BDC3C7;
        border-radius: 3px;
    }

    QListWidget::item, QTreeWidget::item, QListView::item, QTreeView::item {
        padding: 4px;
        border-bottom: 1px solid #ECF0F1;
    }

    QListWidget::item:selected, QTreeWidget::item:selected, QListView::item:selected, QTreeView::item:selected {
        background-color: #3498DB;
        color: white;
    }

    QListWidget::item:hover, QTreeWidget::item:hover, QListView::item:hover, QTreeView::item:hover {
        background-color: #E3F2FD;
    }

    /* Вкладки */
    QTabWidget::pane {
        border: 1px solid #BDC3C7;
        border-radius: 3px;
        top: -1px;
    }

    QTabBar::tab {
        background-color: #ECF0F1;
        border: 1px solid #BDC3C7;
        border-bottom: none;
        border-top-left-radius: 3px;
        border-top-right-radius: 3px;
        padding: 5px 10px;
        margin-right: 2px;
    }

    QTabBar::tab:selected {
        background-color: #F5F7FA;
        border-bottom: 1px solid #F5F7FA;
    }

    QTabBar::tab:hover:!selected {
        background-color: #D6EAF8;
    }

    /* Спліттери */
    QSplitter::handle {
        background-color: #BDC3C7;
        width: 1px;
        height: 1px;
    }

    QSplitter::handle:horizontal {
        width: 5px;
    }

    QSplitter::handle:vertical {
        height: 5px;
    }

    QSplitter::handle:hover {
        background-color: #3498DB;
    }

    /* Панелі інструментів і фрейми */
    QToolBar, QFrame {
        border: 1px solid #BDC3C7;
        border-radius: 3px;
    }

    /* Спеціальні стилі для інспектора */
    QFrame#inspector_frame {
        background-color: #EFF0F1;
        border-right: 1px solid #BDC3C7;
    }

    /* Спеціальні стилі для центральної області */
    QFrame#central_area {
        background-color: white;
    }

    /* Статусний рядок */
    QStatusBar {
        background-color: #ECF0F1;
        color: #34495E;
        border-top: 1px solid #BDC3C7;
    }

    /* Графічний вигляд */
    GraphicsView {
        background-color: white;
        border: 1px solid #BDC3C7;
    }

    /* Меню */
    QMenuBar {
        background-color: #ECF0F1;
        border-bottom: 1px solid #BDC3C7;
    }

    QMenuBar::item {
        background-color: transparent;
        padding: 6px 10px;
    }

    QMenuBar::item:selected {
        background-color: #D6EAF8;
        border-radius: 3px;
    }

    QMenu {
        background-color: white;
        border: 1px solid #BDC3C7;
        border-radius: 3px;
    }

    QMenu::item {
        padding: 5px 25px 5px 20px;
    }

    QMenu::item:selected {
        background-color: #D6EAF8;
    }

    /* Смуги прокрутки */
    QScrollBar:vertical {
        border: none;
        background-color: #F5F7FA;
        width: 10px;
        margin: 0px;
    }

    QScrollBar::handle:vertical {
        background-color: #BDC3C7;
        border-radius: 5px;
        min-height: 20px;
    }

    QScrollBar::handle:vertical:hover {
        background-color: #95A5A6;
    }

    QScrollBar:horizontal {
        border: none;
        background-color: #F5F7FA;
        height: 10px;
        margin: 0px;
    }

    QScrollBar::handle:horizontal {
        background-color: #BDC3C7;
        border-radius: 5px;
        min-width: 20px;
    }

    QScrollBar::handle:horizontal:hover {
        background-color: #95A5A6;
    }

    /* Розділювачі (роздільники між елементами) */
    QFrame[frameShape="4"], QFrame[frameShape="HLine"] {
        background-color: #BDC3C7;
        border: none;
        max-height: 1px;
    }

    QFrame[frameShape="5"], QFrame[frameShape="VLine"] {
        background-color: #BDC3C7;
        border: none;
        max-width: 1px;
    }

    /* Спеціальні стилі для статусів експерименту */
    QLabel#status_value[status="finished"] {
        color: #27AE60;
        font-weight: bold;
    }

    QLabel#status_value[status="not_started"] {
        color: #E74C3C;
        font-weight: bold;
    }

    QLabel#status_value[status="running"] {
        color: #F39C12;
        font-weight: bold;
    }

    /* Стиль для заголовку інспектора */
    QLabel#inspector_header {
        font-size: 14pt;
        font-weight: bold;
        color: #2C3E50;
        padding-bottom: 8px;
    }
    .best_indicator {
        background-color: #B4FFB4;
        border: 1px solid #2ECC71;
    }
    
    .worst_indicator {
        background-color: #FFB4B4;
        border: 1px solid #E74C3C;
    }
    """

    app.setStyleSheet(stylesheet)

    def traverse_widgets(widget):

        if isinstance(widget, QObject):
            if hasattr(widget, 'objectName') and widget.objectName():
                pass
            elif hasattr(widget, 'text'):
                text = widget.text() if callable(widget.text) else ''

                if hasattr(widget, 'setObjectName'):
                    if text == "Розпочати" or text == "Start":
                        widget.setObjectName("start_button")
                    elif text == "Оцінити" or text == "Evaluate":
                        widget.setObjectName("evaluate_button")
                    elif text == "Успадкувати" or text == "Inherit":
                        widget.setObjectName("inherit_button")
                    elif text == "Зберегти модель" or text == "Save model":
                        widget.setObjectName("save_button")
                    elif text == "Створити експеримент" or text == "New experiment":
                        widget.setObjectName("new_experiment_button")

            if hasattr(widget, 'text') and callable(widget.text):
                if widget.text() == "Завершено":
                    widget.setProperty("status", "finished")
                elif widget.text() == "Не запущено":
                    widget.setProperty("status", "not_started")
                elif widget.text() == "Виконується":
                    widget.setProperty("status", "running")

                if hasattr(widget, 'font') and widget.font().pointSize() > 12:
                    widget.setProperty("heading", "true")

        if hasattr(widget, 'children'):
            for child in widget.children():
                traverse_widgets(child)

    for widget in app.allWidgets():
        if isinstance(widget, QMainWindow):
            traverse_widgets(widget)

def update_status_style(label, status):

    label.setProperty("status", status)
    label.style().unpolish(label)
    label.style().polish(label)
    label.update()

