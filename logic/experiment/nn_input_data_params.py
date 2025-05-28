from typing import Dict, Any

from project.logic.experiment.input_data_params import InputDataParams


class NeuralNetInputDataParams(InputDataParams):

    def __init__(self):
        super().__init__()

        # Шляхи до файлів моделі
        self.model_file_path = ''
        self.weights_file_path = ''
        self.model_config_path = ''

        # Додаткові параметри для нейронних мереж
        self.load_type = ''

        self.text_directory = ''
        self.image_directory = ''

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            'model_file_path': self.model_file_path,
            'weights_file_path': self.weights_file_path,
            'model_config_path': self.model_config_path,
            'load_type': self.load_type,
        })

        return data
