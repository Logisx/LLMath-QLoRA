# -*- coding: utf-8 -*-
import os
from pathlib import Path
from src.logging import logger

from src.data.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from src.data.pipeline.stage_02_data_preprocessing import DataPreprocessingPipeline


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    try:
        logger.info('>>>>> Data Ingestion started <<<<<')

        data_ingestion_pipeline = DataIngestionPipeline()
        data_ingestion_pipeline.main()

        logger.info('>>>>> Data Ingestion completed <<<<<')

    except Exception as e:
        logger.exception(e)
        raise e
    

    try:
        logger.info('>>>>> Data Preprocessing started <<<<<')

        data_preprocessing_pipeline = DataPreprocessingPipeline()
        data_preprocessing_pipeline.main()

        logger.info('>>>>> Data Preprocessing completed <<<<<')
        
    except Exception as e:
        logger.exception(e)
        raise e
    

if __name__ == '__main__':
    
    os.chdir("../../")

    main()
