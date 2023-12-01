# -*- coding: utf-8 -*-
import os
from pathlib import Path
from src.logging import logger

from src.models.pipeline.stage_01_model_training import ModelTrainingPipeline


def main():
    """ Run model training pipeline """
    try:
        logger.info('>>>>> Model Training pipeline started <<<<<')

        model_training_pipeline = ModelTrainingPipeline()
        model_training_pipeline.main()

        logger.info('>>>>> Model Training pipeline completed <<<<<')

    except Exception as e:
        logger.exception(e)
        raise e


if __name__ == '__main__':
    
    os.chdir("../../")

    main()
