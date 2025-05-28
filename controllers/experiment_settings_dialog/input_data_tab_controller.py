import os

from PyQt5.QtWidgets import QMessageBox

from project.controllers.experiment_settings_dialog.tab_controller import TabController
from project.logic.experiment.experiment import Experiment


class InputDataTabController(TabController):
    """Controller for input data parameters tab"""

    def __init__(self, experiment: Experiment, view):
        super().__init__(experiment, view)
        self.input_data_params = self.experiment.input_data_params
        self.input_data_params.current_task = self.experiment.task
        self.connect_signals()
        self.init_view()

    def connect_signals(self):
        # Data processing mode signals
        self.view.single_file_radio.toggled.connect(self.on_mode_changed)
        self.view.multi_files_radio.toggled.connect(self.on_mode_changed)

        # File selection signals
        self.view.single_file_btn.clicked.connect(self.browse_single_file)
        self.view.x_train_file_btn.clicked.connect(lambda: self.browse_file('x_train'))
        self.view.y_train_file_btn.clicked.connect(lambda: self.browse_file('y_train'))
        self.view.x_test_file_btn.clicked.connect(lambda: self.browse_file('x_test'))
        self.view.y_test_file_btn.clicked.connect(lambda: self.browse_file('y_test'))

        # File path change signals
        self.view.single_file_path.textChanged.connect(lambda text: self.on_file_path_changed(text, 'single'))
        self.view.x_train_file_path.textChanged.connect(lambda text: self.on_file_path_changed(text, 'x_train'))
        self.view.y_train_file_path.textChanged.connect(lambda text: self.on_file_path_changed(text, 'y_train'))
        self.view.x_test_file_path.textChanged.connect(lambda text: self.on_file_path_changed(text, 'x_test'))
        self.view.y_test_file_path.textChanged.connect(lambda text: self.on_file_path_changed(text, 'y_test'))

        # Train/test split signals
        self.view.train_percent.valueChanged.connect(self.on_train_percent_changed)

        # Target variable signals
        self.view.target_combo.currentTextChanged.connect(self.on_target_changed)

        # Manual encoding checkboxes
        self.view.single_manual_encoding_checkbox.toggled.connect(self.on_manual_encoding_toggled)
        self.view.multi_manual_encoding_checkbox.toggled.connect(self.on_manual_encoding_toggled)

        # Single file encoding
        self.view.single_encoding_combo.currentTextChanged.connect(
            lambda text: self.on_encoding_changed('single', text))

        # Multi-file encoding and separators
        self.view.multi_encoding_combo.currentTextChanged.connect(
            lambda text: self.on_multi_encoding_changed(text))
        self.view.multi_separator_combo.currentTextChanged.connect(
            lambda text: self.on_multi_separator_changed(text))

        # Single file separator
        self.view.single_separator_combo.currentTextChanged.connect(
            lambda text: self.on_separator_changed('single', text))

        # Random seed
        self.view.seed_spinbox.valueChanged.connect(lambda value: setattr(self.input_data_params, 'seed', value))

        # Categorical encoding mode
        self.view.one_hot_radio.clicked.connect(self.on_categorical_encoding_changed)
        self.view.to_categorical_radio.clicked.connect(self.on_categorical_encoding_changed)

    def init_view(self):
        # Set single file mode as default
        self.view.single_file_radio.setChecked(True)

        # Update target field visibility based on task type
        supervised_learning = not self.input_data_params.is_target_not_required()
        self.view.update_ui_state(single_file_mode=True, supervised_learning=supervised_learning)
        self.view.update_target_field_visibility(supervised_learning)

        # Initialize encoding fields state
        self.view.update_encoding_fields_visibility()
        self.view.load_input_data_params(self.input_data_params)

    def on_mode_changed(self):
        single_file_mode = self.view.single_file_radio.isChecked()
        self.input_data_params.mode = 'single_file' if single_file_mode else 'multi_files'

        supervised_learning = not self.input_data_params.is_target_not_required()
        self.view.update_ui_state(single_file_mode, supervised_learning)

    def on_train_percent_changed(self, value):
        self.input_data_params.train_percent = value
        self.input_data_params.test_percent = 100 - value
        self.view.update_test_percent(value)

    def on_target_changed(self, value):
        self.input_data_params.target_variable = value

    def on_manual_encoding_toggled(self):
        # Update encoding fields visibility
        self.view.update_encoding_fields_visibility()

    def on_encoding_changed(self, file_type, value):
        if file_type == 'single':
            self.input_data_params.file_encoding = value
            if self.input_data_params.single_file_path:
                self.load_column_names(self.input_data_params.single_file_path)

    def on_multi_encoding_changed(self, value):
        # Update encoding for all files in multi_files mode
        self.input_data_params.file_encoding = value

    def on_separator_changed(self, file_type, value):
        if file_type == 'single':
            self.input_data_params.file_separator = value
            if self.input_data_params.single_file_path and os.path.splitext(self.input_data_params.single_file_path)[
                1].lower() == '.csv':
                self.load_column_names(self.input_data_params.single_file_path)

    def on_categorical_encoding_changed(self):
        self.experiment.input_data_params.categorical_encoding = self.view.get_categorical_encoding_method()

    def on_multi_separator_changed(self, value):
        # Update separator for all files in multi_files mode
        self.input_data_params.file_separator = value

    def on_file_path_changed(self, path, file_type):
        """Handler for file path changes (shows/hides separator fields)"""
        self.view.show_separator_fields(path, file_type)

        # Update corresponding field in model
        if file_type == 'single':
            self.input_data_params.single_file_path = path
            # Auto-detect encoding and separator if manual mode is disabled
            if not self.view.single_manual_encoding_checkbox.isChecked() and path:
                self.auto_detect_format(path, 'single')
        elif file_type == 'x_train':
            self.input_data_params.x_train_file_path = path
            if not self.view.multi_manual_encoding_checkbox.isChecked() and path:
                self.auto_detect_format(path, 'multi')
        elif file_type == 'y_train':
            self.input_data_params.y_train_file_path = path
        elif file_type == 'x_test':
            self.input_data_params.x_test_file_path = path
        elif file_type == 'y_test':
            self.input_data_params.y_test_file_path = path

    def browse_single_file(self):
        file, _ = self.view.get_file_dialog(
            "Вибрати файл даних",  # "Select data file"
            "Усі файли (*.csv *.xlsx *.xls *.json *.parquet);;CSV файли (*.csv);;Excel файли (*.xlsx *.xls);;JSON файли (*.json);;Parquet файли (*.parquet)"
            # "All files..."
        )
        if file:
            self.input_data_params.single_file_path = file
            self.view.single_file_path.setText(file)

            # Auto-detect format if manual mode is disabled
            if not self.view.single_manual_encoding_checkbox.isChecked():
                success = self.auto_detect_format(file, 'single')
                if success:
                    self.load_column_names(file)
            else:
                self.load_column_names(file)

    def browse_file(self, file_type):
        title_map = {
            'x_train': "Вибрати файл для тренувальних даних (X_train)",  # "Select training data file (X_train)"
            'y_train': "Вибрати файл для тренувальних міток (y_train)",  # "Select training labels file (y_train)"
            'x_test': "Вибрати файл для тестових даних (X_test)",  # "Select test data file (X_test)"
            'y_test': "Вибрати файл для тестових міток (y_test)"  # "Select test labels file (y_test)"
        }
        file, _ = self.view.get_file_dialog(
            title_map.get(file_type, "Вибрати файл"),  # "Select file"
            "Усі файли (*.csv *.xlsx *.xls *.json *.parquet);;CSV файли (*.csv);;Excel файли (*.xlsx *.xls);;JSON файли (*.json);;Parquet файли (*.parquet)"
            # "All files..."
        )
        if file:
            if file_type == 'x_train':
                self.input_data_params.x_train_file_path = file
                self.view.x_train_file_path.setText(file)
                # Auto-detect format for all files in multi mode
                if not self.view.multi_manual_encoding_checkbox.isChecked():
                    self.auto_detect_format(file, 'multi')
            elif file_type == 'y_train':
                self.input_data_params.y_train_file_path = file
                self.view.y_train_file_path.setText(file)
            elif file_type == 'x_test':
                self.input_data_params.x_test_file_path = file
                self.view.x_test_file_path.setText(file)
            elif file_type == 'y_test':
                self.input_data_params.y_test_file_path = file
                self.view.y_test_file_path.setText(file)

    def auto_detect_format(self, file_path, mode='single'):
        """Auto-detects CSV file encoding and separator"""
        ext = os.path.splitext(file_path)[1].lower()
        if ext != '.csv':
            return False

        encodings = ['utf-8', 'cp1251', 'latin-1', 'iso-8859-1', 'ascii']
        separators = [',', ';', '\t', '|', ' ']

        import pandas as pd

        for encoding in encodings:
            for separator in separators:
                try:
                    df = pd.read_csv(file_path, nrows=5, encoding=encoding, sep=separator)
                    if len(df.columns) > 1:  # Check if multiple columns were successfully recognized
                        # Update model and UI values based on mode
                        if mode == 'single':
                            self.input_data_params.file_encoding = encoding
                            self.view.single_encoding_combo.setCurrentText(encoding)
                            self.input_data_params.file_separator = self.get_separator_display(separator)
                            self.view.single_separator_combo.setCurrentText(self.get_separator_display(separator))
                        else:  # multi
                            # Update values for all files in multi_files mode
                            self.input_data_params.file_encoding = encoding
                            self.view.multi_encoding_combo.setCurrentText(encoding)

                            self.input_data_params.file_separator = self.get_separator_display(separator)
                            self.view.multi_separator_combo.setCurrentText(self.get_separator_display(separator))

                        # For single file, update target variable dropdown
                        if mode == 'single':
                            self.view.target_combo.clear()
                            self.view.target_combo.addItems([str(col) for col in df.columns.tolist()])
                            if df.columns.tolist():
                                self.input_data_params.target_variable = str(df.columns.tolist()[0])

                        return True
                except Exception:
                    continue

        # Show warning if auto-detection failed
        QMessageBox.warning(
            self.view,
            "Не вдалося визначити формат",  # "Failed to detect format"
            "Не вдалося автоматично визначити кодування та роздільник файлу. Будь ласка, виберіть їх вручну."
            # "Could not auto-detect file encoding and separator. Please select them manually."
        )

        # Enable manual settings checkbox
        if mode == 'single':
            self.view.single_manual_encoding_checkbox.setChecked(True)
        else:
            self.view.multi_manual_encoding_checkbox.setChecked(True)

        return False

    def load_column_names(self, file_path):
        """Loads column names from file and sets them in dropdown"""
        try:
            # Clear current list
            self.view.target_combo.clear()

            # Determine file format and load headers
            ext = os.path.splitext(file_path)[1].lower()
            column_names = []

            if ext == '.csv':
                try:
                    import pandas as pd
                    # Use selected encoding and separator
                    df = pd.read_csv(
                        file_path,
                        nrows=1,
                        encoding=self.input_data_params.file_encoding,
                        sep=self.convert_separator(self.input_data_params.file_separator)
                    )
                    column_names = [str(col) for col in df.columns.tolist()]
                except Exception as e:
                    print(f"CSV read error: {e}")

            elif ext in ['.xlsx', '.xls']:
                try:
                    import pandas as pd
                    df = pd.read_excel(file_path, nrows=1, header=0)
                    column_names = [str(col) for col in df.columns.tolist()]

                    if all(col.startswith('Unnamed:') or col.startswith('Column') for col in column_names):
                        df = pd.read_excel(file_path, nrows=1, header=None)
                        column_names = [f"Column {i + 1}" for i in range(len(df.columns))]
                        first_row_values = [str(val) for val in df.iloc[0].tolist()]
                        if any(val and not val.isspace() for val in first_row_values):
                            column_names = first_row_values
                except Exception as e:
                    print(f"Excel read error: {e}")

            elif ext == '.json':
                try:
                    import pandas as pd
                    df = pd.read_json(file_path, encoding=self.input_data_params.file_encoding)
                    column_names = [str(col) for col in df.columns.tolist()]
                except Exception as e:
                    print(f"JSON read error: {e}")
                    if not self.view.single_manual_encoding_checkbox.isChecked():
                        self.auto_detect_format(file_path, 'single')

            elif ext == '.parquet':
                try:
                    import pandas as pd
                    df = pd.read_parquet(file_path)
                    column_names = [str(col) for col in df.columns.tolist()]
                except Exception as e:
                    print(f"Parquet read error: {e}")

            # Add column names to dropdown
            if column_names:
                self.view.target_combo.addItems(column_names)
                # Set first column as default target variable
                if column_names:
                    self.input_data_params.target_variable = column_names[0]

        except Exception as e:
            print(f"Error reading file headers: {e}")
            QMessageBox.warning(
                self.view,
                "Помилка читання файлу",  # "File read error"
                f"Не вдалося прочитати заголовки файлу. Перевірте формат, кодування та роздільник.\nПомилка: {str(e)}"
                # f"Failed to read file headers. Check format, encoding and separator.\nError: {str(e)}"
            )

    def convert_separator(self, separator):
        """Converts displayed separator to actual pandas separator"""
        if separator == "\\t":
            return "\t"
        return separator

    def get_separator_display(self, separator):
        """Converts actual separator to displayed separator for UI"""
        if separator == "\t":
            return "\\t"
        return separator

    def update_model_from_view(self):
        try:
            self.input_data_params.mode = 'single_file' if self.view.single_file_radio.isChecked() else 'multi_files'

            # Single file settings
            self.input_data_params.single_file_path = self.view.single_file_path.text()
            self.input_data_params.file_encoding = self.view.single_encoding_combo.currentText()
            self.input_data_params.file_separator = self.view.single_separator_combo.currentText()

            # Multi-file settings - use common settings
            if self.input_data_params.mode == 'multi_files':
                common_encoding = self.view.multi_encoding_combo.currentText()
                common_separator = self.view.multi_separator_combo.currentText()

                # Set common settings for all files
                self.input_data_params.file_encoding = common_encoding
                self.input_data_params.file_separator = common_separator

            # Multi-file paths
            self.input_data_params.x_train_file_path = self.view.x_train_file_path.text()
            self.input_data_params.y_train_file_path = self.view.y_train_file_path.text()
            self.input_data_params.x_test_file_path = self.view.x_test_file_path.text()
            self.input_data_params.y_test_file_path = self.view.y_test_file_path.text()

            # Split parameters
            self.input_data_params.train_percent = self.view.train_percent.value()
            self.input_data_params.test_percent = 100 - self.view.train_percent.value()
            self.input_data_params.seed = self.view.seed_spinbox.value()

            # Target variable (only if visible and supervised learning)
            if not self.input_data_params.is_target_not_required() and self.view.target_combo.isVisible():
                self.input_data_params.target_variable = self.view.target_combo.currentText()

            self.input_data_params.categorical_encoding = self.view.get_categorical_encoding_method()
        except Exception as e:
            QMessageBox.warning(self.view, "Хибні параметри",
                                f"Виникла помилка у налаштуваннях\n {e}")  # "Invalid parameters", "Error in settings\n {e}"

    def get_input_params(self):
        self.update_model_from_view()
        return self.input_data_params.to_dict()