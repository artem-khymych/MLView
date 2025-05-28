class ParameterEditorController:
    """Controller for ParameterEditorWidget"""

    def __init__(self):
        self.view = None

    def set_view(self, view):
        self.view = view

    def show(self, params_dict):
        """Loads parameters into the widget"""
        if self.view:
            self.view.populate_table(params_dict)

    def on_parameters_changed(self, updated_params):
        """Handler for parameter change event"""
        # Additional logic can be added here before further processing
        print("Controller: Parameters updated:", updated_params)
        return updated_params

    def get_current_parameters(self):
        """Returns current parameters from the widget"""
        if self.view:
            return self.view.get_current_parameters()
        return {}