# -*- coding: utf-8 -*-
from src.logging import logger

from src.models.pipeline.prediction import ModelPredictionPipeline
from src.models.configuration import ConfigurationManager


class PredictionPipeline:
    def __init__(self):
        pass
    
    
    def predict(self, query):
        """ Run model prediction pipeline """
        try:
            model_prediction_pipeline = ModelPredictionPipeline()
            output = model_prediction_pipeline.main(query)
            return output

        except Exception as e:
            logger.exception(e)
            raise e
    

