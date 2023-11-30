# -*- coding: utf-8 -*-
import os
from pathlib import Path
from src.logging import logger

from src.data.configuration import ConfigurationManager

from src.data.components.data_ingestion import DataIngestion


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger.info('>>>>> Data Ingestion started <<<<<')
    
    config = ConfigurationManager()
    data_ingestion_config = config.get_data_ingestion_config()
    data_ingestion = DataIngestion(config=data_ingestion_config)
    data_ingestion.download_file()
    data_ingestion.extract_zip_file()

    logger.info('>>>>> Data Ingestion completed <<<<<')

if __name__ == '__main__':
    
    os.chdir("../../")

    main()
