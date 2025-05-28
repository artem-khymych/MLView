import os

from PyQt5.QtWidgets import (QFileDialog, QGroupBox, QRadioButton, QHBoxLayout, QSpinBox, QLabel,
                             QVBoxLayout, QComboBox, QPushButton, QLineEdit, QWidget, QCheckBox)


class InputDataTabWidget(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):

        self.main_layout = QVBoxLayout()

        self.create_data_mode_group()
        self.main_layout.addWidget(self.data_mode_group)

        self.single_file_group = QGroupBox("Налаштування для одного файлу")
        single_file_layout = QVBoxLayout()

        self.single_file_layout = QHBoxLayout()
        self.single_file_label = QLabel("Шлях до файлу:")
        self.single_file_path = QLineEdit()
        self.single_file_btn = QPushButton("Обрати...")
        self.single_file_layout.addWidget(self.single_file_label)
        self.single_file_layout.addWidget(self.single_file_path)
        self.single_file_layout.addWidget(self.single_file_btn)
        single_file_layout.addLayout(self.single_file_layout)

        # Group for encoding with checkbox
        self.single_encoding_group = QHBoxLayout()
        self.single_manual_encoding_checkbox = QCheckBox("Ручне налаштування кодування")
        self.single_encoding_group.addWidget(self.single_manual_encoding_checkbox)
        single_file_layout.addLayout(self.single_encoding_group)

        self.single_encoding_layout = QHBoxLayout()
        self.single_encoding_label = QLabel("Кодування:")
        self.single_encoding_combo = QComboBox()
        self.single_encoding_combo.addItems(["utf-8", "cp1251", "latin-1", "iso-8859-1", "ascii"])
        self.single_encoding_layout.addWidget(self.single_encoding_label)
        self.single_encoding_layout.addWidget(self.single_encoding_combo)
        single_file_layout.addLayout(self.single_encoding_layout)

        self.single_separator_layout = QHBoxLayout()
        self.single_separator_label = QLabel("Роздільник CSV:")
        self.single_separator_combo = QComboBox()
        self.single_separator_combo.addItems([",", ";", "\\t", "|", " "])
        self.single_separator_layout.addWidget(self.single_separator_label)
        self.single_separator_layout.addWidget(self.single_separator_combo)
        single_file_layout.addLayout(self.single_separator_layout)

        self.target_layout = QHBoxLayout()
        self.target_label = QLabel("Цільова змінна:")
        self.target_combo = QComboBox()
        self.target_layout.addWidget(self.target_label)
        self.target_layout.addWidget(self.target_combo)
        single_file_layout.addLayout(self.target_layout)

        self.single_file_group.setLayout(single_file_layout)
        self.main_layout.addWidget(self.single_file_group)

        self.multi_files_group = QGroupBox("Налаштування для декількох файлів")
        multi_files_layout = QVBoxLayout()

        # Group for common settings in multiple files mode
        self.multi_common_settings_group = QGroupBox("Спільні налаштування для всіх файлів")
        multi_common_settings_layout = QVBoxLayout()

        # Checkbox for manual encoding setup (multiple files mode)
        self.multi_manual_encoding_checkbox = QCheckBox("Ручне налаштування кодування та роздільника")
        multi_common_settings_layout.addWidget(self.multi_manual_encoding_checkbox)

        # Common encoding for all files
        self.multi_encoding_layout = QHBoxLayout()
        self.multi_encoding_label = QLabel("Кодування для всіх файлів:")
        self.multi_encoding_combo = QComboBox()
        self.multi_encoding_combo.addItems(["utf-8", "cp1251", "latin-1", "iso-8859-1", "ascii"])
        self.multi_encoding_layout.addWidget(self.multi_encoding_label)
        self.multi_encoding_layout.addWidget(self.multi_encoding_combo)
        multi_common_settings_layout.addLayout(self.multi_encoding_layout)

        # Common separator for all files
        self.multi_separator_layout = QHBoxLayout()
        self.multi_separator_label = QLabel("Роздільник CSV для всіх файлів:")
        self.multi_separator_combo = QComboBox()
        self.multi_separator_combo.addItems([",", ";", "\\t", "|", " "])
        self.multi_separator_layout.addWidget(self.multi_separator_label)
        self.multi_separator_layout.addWidget(self.multi_separator_combo)
        multi_common_settings_layout.addLayout(self.multi_separator_layout)

        self.multi_common_settings_group.setLayout(multi_common_settings_layout)
        multi_files_layout.addWidget(self.multi_common_settings_group)

        # X_train section
        self.x_train_group = QGroupBox("Тренувальні дані для навчання (X_train)")
        x_train_layout = QVBoxLayout()

        x_train_path_layout = QHBoxLayout()
        self.x_train_file_label = QLabel("Шлях до файлу:")
        self.x_train_file_path = QLineEdit()
        self.x_train_file_btn = QPushButton("Обрати...")
        x_train_path_layout.addWidget(self.x_train_file_label)
        x_train_path_layout.addWidget(self.x_train_file_path)
        x_train_path_layout.addWidget(self.x_train_file_btn)
        x_train_layout.addLayout(x_train_path_layout)

        self.x_train_group.setLayout(x_train_layout)
        multi_files_layout.addWidget(self.x_train_group)

        # Y_train section (for supervised tasks)
        self.y_train_group = QGroupBox("Тренувальні дані для тестування (y_train)")
        y_train_layout = QVBoxLayout()

        y_train_path_layout = QHBoxLayout()
        self.y_train_file_label = QLabel("Шлях до файлу:")
        self.y_train_file_path = QLineEdit()
        self.y_train_file_btn = QPushButton("Обрати...")
        y_train_path_layout.addWidget(self.y_train_file_label)
        y_train_path_layout.addWidget(self.y_train_file_path)
        y_train_path_layout.addWidget(self.y_train_file_btn)
        y_train_layout.addLayout(y_train_path_layout)

        self.y_train_group.setLayout(y_train_layout)
        multi_files_layout.addWidget(self.y_train_group)

        # X_test section
        self.x_test_group = QGroupBox("Тестові дані для навчання (X_test)")
        x_test_layout = QVBoxLayout()

        x_test_path_layout = QHBoxLayout()
        self.x_test_file_label = QLabel("Шлях до файлу:")
        self.x_test_file_path = QLineEdit()
        self.x_test_file_btn = QPushButton("Обрати...")
        x_test_path_layout.addWidget(self.x_test_file_label)
        x_test_path_layout.addWidget(self.x_test_file_path)
        x_test_path_layout.addWidget(self.x_test_file_btn)
        x_test_layout.addLayout(x_test_path_layout)

        self.x_test_group.setLayout(x_test_layout)
        multi_files_layout.addWidget(self.x_test_group)

        # Y_test section (for supervised tasks)
        self.y_test_group = QGroupBox("Тестові дані для тестування (y_test)")
        y_test_layout = QVBoxLayout()

        y_test_path_layout = QHBoxLayout()
        self.y_test_file_label = QLabel("Шлях до файлу:")
        self.y_test_file_path = QLineEdit()
        self.y_test_file_btn = QPushButton("Обрати...")
        y_test_path_layout.addWidget(self.y_test_file_label)
        y_test_path_layout.addWidget(self.y_test_file_path)
        y_test_path_layout.addWidget(self.y_test_file_btn)
        y_test_layout.addLayout(y_test_path_layout)

        self.y_test_group.setLayout(y_test_layout)
        multi_files_layout.addWidget(self.y_test_group)

        self.multi_files_group.setLayout(multi_files_layout)
        self.main_layout.addWidget(self.multi_files_group)

        self.split_group = QGroupBox("Параметри розбиття")
        split_layout = QVBoxLayout()

        split_ratio_layout = QHBoxLayout()
        split_ratio_layout.addWidget(QLabel("Відсоток для тренування:"))

        self.train_percent = QSpinBox()
        self.train_percent.setRange(10, 90)
        self.train_percent.setValue(80)
        self.train_percent.setSuffix("%")
        split_ratio_layout.addWidget(self.train_percent)

        split_ratio_layout.addWidget(QLabel("Тестування:"))
        self.test_percent = QLabel("20%")
        split_ratio_layout.addWidget(self.test_percent)

        split_layout.addLayout(split_ratio_layout)

        seed_layout = QHBoxLayout()
        seed_layout.addWidget(QLabel("Seed значення:"))
        self.seed_spinbox = QSpinBox()
        self.seed_spinbox.setRange(0, 999999)
        self.seed_spinbox.setValue(42)
        seed_layout.addWidget(self.seed_spinbox)
        split_layout.addLayout(seed_layout)

        self.split_group.setLayout(split_layout)
        self.main_layout.addWidget(self.split_group)


        self.setLayout(self.main_layout)

        # Initialize encoding state
        self.single_manual_encoding_checkbox.setChecked(False)
        self.multi_manual_encoding_checkbox.setChecked(False)
        self.update_encoding_fields_visibility()

        # Add group for categorical variables processing
        self.categorical_encoding_group = QGroupBox("Обробка категоріальних змінних")
        categorical_encoding_layout = QVBoxLayout()

        # Instruction or explanation for user
        categorical_info_label = QLabel("Виберіть метод кодування категоріальних змінних:")
        categorical_encoding_layout.addWidget(categorical_info_label)

        # Create horizontal container for radio buttons
        encoding_options_layout = QHBoxLayout()

        # Create radio buttons for encoding method selection
        self.one_hot_radio = QRadioButton("One-Hot Encoding")

        self.to_categorical_radio = QRadioButton("Присвоєння числових міток")

        # Add radio buttons to horizontal container
        encoding_options_layout.addWidget(self.one_hot_radio)
        encoding_options_layout.addWidget(self.to_categorical_radio)

        # Add container with radio buttons to main group container
        categorical_encoding_layout.addLayout(encoding_options_layout)

        # Select one-hot encoding by default
        self.one_hot_radio.setChecked(True)

        # Set layout for group
        self.categorical_encoding_group.setLayout(categorical_encoding_layout)

        # Add group to main layout
        self.main_layout.addWidget(self.categorical_encoding_group)

        self.main_layout.addStretch()
        self.setLayout(self.main_layout)

        # Initialize encoding state
        self.single_manual_encoding_checkbox.setChecked(False)
        self.multi_manual_encoding_checkbox.setChecked(False)
        self.update_encoding_fields_visibility()

        # Set separator fields visibility
        self.show_separator_fields(self.single_file_path.text(), 'single')


    def load_input_data_params(self, input_data_params):
        """
        Fills widget fields with data from InputDataParams object

        Parameters:
        input_data_params (InputDataParams): Object with input data parameters
        """
        if not input_data_params.is_filled():
            return

        # Set mode
        if input_data_params.mode == 'single_file':
            self.single_file_radio.setChecked(True)
            self.update_ui_state(single_file_mode=True,
                                 supervised_learning=not input_data_params.is_target_not_required())

            # Fill file path
            if input_data_params.single_file_path:
                self.single_file_path.setText(input_data_params.single_file_path)

            # Split parameters
            self.train_percent.setValue(input_data_params.train_percent)
            self.test_percent.setText(f"{input_data_params.test_percent}%")
            self.seed_spinbox.setValue(input_data_params.seed)

            # If file has target variable and it's required for current task
            if input_data_params.target_variable and not input_data_params.is_target_not_required():
                self.update_target_field_visibility(True)
                # Need to first fill combobox with columns from file
                # then set selected target_variable
                # This code should be called after file loading
                index = self.target_combo.findText(input_data_params.target_variable)
                if index >= 0:
                    self.target_combo.setCurrentIndex(index)
            else:
                self.update_target_field_visibility(not input_data_params.is_target_not_required())
        else:
            # Multiple files mode
            self.multi_files_radio.setChecked(True)
            self.update_ui_state(single_file_mode=False,
                                 supervised_learning=not input_data_params.is_target_not_required())

            # Fill file paths
            if input_data_params.x_train_file_path:
                self.x_train_file_path.setText(input_data_params.x_train_file_path)

            if input_data_params.x_test_file_path:
                self.x_test_file_path.setText(input_data_params.x_test_file_path)

            # Add y-files only for supervised tasks
            if not input_data_params.is_target_not_required():
                if input_data_params.y_train_file_path:
                    self.y_train_file_path.setText(input_data_params.y_train_file_path)

                if input_data_params.y_test_file_path:
                    self.y_test_file_path.setText(input_data_params.y_test_file_path)

        # Encoding settings for both modes
        if input_data_params.file_encoding != 'utf-8' or input_data_params.file_separator != ',':
            if input_data_params.mode == 'single_file':
                self.single_manual_encoding_checkbox.setChecked(True)
                index = self.single_encoding_combo.findText(input_data_params.file_encoding)
                if index >= 0:
                    self.single_encoding_combo.setCurrentIndex(index)

                index = self.single_separator_combo.findText(input_data_params.file_separator)
                if index >= 0:
                    self.single_separator_combo.setCurrentIndex(index)
            else:
                self.multi_manual_encoding_checkbox.setChecked(True)
                index = self.multi_encoding_combo.findText(input_data_params.file_encoding)
                if index >= 0:
                    self.multi_encoding_combo.setCurrentIndex(index)

                index = self.multi_separator_combo.findText(input_data_params.file_separator)
                if index >= 0:
                    self.multi_separator_combo.setCurrentIndex(index)

        # Update encoding fields visibility
        self.update_encoding_fields_visibility()

        # Categorical variables processing settings
        if input_data_params.categorical_encoding == 'one-hot':
            self.one_hot_radio.setChecked(True)
        elif input_data_params.categorical_encoding == 'to_categorical':
            self.to_categorical_radio.setChecked(True)

    def create_data_mode_group(self):
        self.data_mode_group = QGroupBox("Режим роботи з даними")
        data_mode_layout = QHBoxLayout()

        self.single_file_radio = QRadioButton("Один файл (розбити на тренувальний та тестовий)")
        self.multi_files_radio = QRadioButton("Окремі файли для навчання та тестування")

        data_mode_layout.addWidget(self.single_file_radio)
        data_mode_layout.addWidget(self.multi_files_radio)

        self.data_mode_group.setLayout(data_mode_layout)

    def update_ui_state(self, single_file_mode=True, supervised_learning=True):
        self.single_file_group.setVisible(single_file_mode)
        self.multi_files_group.setVisible(not single_file_mode)
        self.split_group.setVisible(single_file_mode)

        if not single_file_mode:
            self.y_train_group.setVisible(supervised_learning)
            self.y_test_group.setVisible(supervised_learning)

    def update_encoding_fields_visibility(self):
        # For single file
        single_manual_mode = self.single_manual_encoding_checkbox.isChecked()
        self.single_encoding_label.setVisible(single_manual_mode)
        self.single_encoding_combo.setVisible(single_manual_mode)
        self.single_separator_label.setVisible(single_manual_mode and self.is_csv_file(self.single_file_path.text()))
        self.single_separator_combo.setVisible(single_manual_mode and self.is_csv_file(self.single_file_path.text()))

        # For multiple files
        multi_manual_mode = self.multi_manual_encoding_checkbox.isChecked()
        self.multi_encoding_label.setVisible(multi_manual_mode)
        self.multi_encoding_combo.setVisible(multi_manual_mode)
        self.multi_separator_label.setVisible(multi_manual_mode)
        self.multi_separator_combo.setVisible(multi_manual_mode)

    def update_test_percent(self, train_value):
        self.test_percent.setText(f"{100 - train_value}%")

    def get_file_dialog(self, title, file_filter):
        return QFileDialog.getOpenFileName(
            self, title, "", file_filter
        )

    def is_csv_file(self, file_path):
        if not file_path:
            return False
        ext = os.path.splitext(file_path)[1].lower()
        return ext == '.csv'

    def show_separator_fields(self, file_path, file_type):
        is_csv = self.is_csv_file(file_path)

        if file_type == 'single':
            manual_mode = self.single_manual_encoding_checkbox.isChecked()
            self.single_separator_label.setVisible(is_csv and manual_mode)
            self.single_separator_combo.setVisible(is_csv and manual_mode)

    def update_target_field_visibility(self, should_show):
        self.target_label.setVisible(should_show)
        self.target_combo.setVisible(should_show)

    def get_categorical_encoding_method(self):
        if self.one_hot_radio.isChecked():
            return "one-hot"
        if self.to_categorical_radio.isChecked():
            return "to_categorical"
        return "one-hot"