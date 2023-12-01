from src.models.configuration import ConfigurationManager
from src.models.components.model_prediction import ModelPrediction

class ModelPredictionPipeline:
    def __init__(self):
        pass

    def main(self, query):           
        question = query
        config = ConfigurationManager()
        model_prediction_config = config.get_model_prediction_config()
        model_prediction_parameters = config.get_model_prediction_parameters()
        bits_and_bytes_parameters = config.get_bits_and_bytes_params()
        model_prediction = ModelPrediction(config=model_prediction_config, bits_and_bytes_parameters=bits_and_bytes_parameters, params=model_prediction_parameters)
        model_prediction.predict(question)
