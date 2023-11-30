import os
from pathlib import Path
from src.utils.common import read_yaml
from dotenv import find_dotenv, load_dotenv
from src.data.entity import (DataIngestionConfig,
)

_ = load_dotenv(find_dotenv()) # read local .env file

DATA_CONFIG_FILE_PATH = os.environ['DATA_CONFIG_FILE_PATH']


class ConfigurationManager:
    def __init__(self,
                 config_filepath = DATA_CONFIG_FILE_PATH):

        self.config = read_yaml(Path(config_filepath))

    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            hf_dataset_name=config.hf_dataset_name,
            hf_dataset_split=config.hf_dataset_split,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config