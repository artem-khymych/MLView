from project.logic.evaluation.task_register import TaskType
from project.logic.modules import task_names


class InputDataParams:

    def __init__(self):
        self.mode = 'single_file'

        self.single_file_path = ''
        self.x_train_file_path = ''
        self.y_train_file_path = ''
        self.x_test_file_path = ''
        self.y_test_file_path = ''

        self.train_percent = 80
        self.test_percent = 20
        self.seed = 42

        self.target_variable = ''

        self.file_encoding = 'utf-8'

        self.file_separator = ','

        self.current_task = ''

        self.categorical_encoding = 'one-hot'

    def to_dict(self):
        data = {
            'mode': self.mode,
            'current_task': self.current_task
        }

        if self.mode == 'single_file':
            data.update({
                'single_file_path': self.single_file_path,
                'train_percent': self.train_percent,
                'test_percent': self.test_percent,
                'seed': self.seed,
                'file_encoding': self.file_encoding,
                'file_separator': self.file_separator,
                'categorical_encoding': self.categorical_encoding
            })

            if self.target_variable and not self.is_target_not_required():
                data['target_variable'] = self.target_variable
        else:
            if self.is_target_not_required():
                data.update({
                    'x_train_file_path': self.x_train_file_path,
                    'x_test_file_path': self.x_test_file_path,
                    'file_encoding': self.file_encoding,
                    'file_separator': self.file_separator,
                    'categorical_encoding': self.categorical_encoding
                })
            else:
                data.update({
                    'x_train_file_path': self.x_train_file_path,
                    'y_train_file_path': self.y_train_file_path,
                    'x_test_file_path': self.x_test_file_path,
                    'y_test_file_path': self.y_test_file_path,
                    'file_encoding': self.file_encoding,
                    'file_separator': self.file_separator,
                    'categorical_encoding': self.categorical_encoding
                })

        return data

    def is_target_not_required(self):
        return self.current_task in InputDataParams.tasks_without_target

    def is_filled(self):
        if self.mode == "single_file":
            if self.single_file_path != "":
                return True
        else:
            if self.x_test_file_path and self.x_train_file_path and self.y_train_file_path and self.y_test_file_path != "":
                return True
        return False

    tasks_without_target = [
        task_names.CLUSTERING,
        task_names.DIMENSIONALITY_REDUCTION,
        task_names.ANOMALY_DETECTION,
        task_names.DENSITY_ESTIMATION,
        TaskType.DIMENSIONALITY_REDUCTION,
        TaskType.ANOMALY_DETECTION
    ]
