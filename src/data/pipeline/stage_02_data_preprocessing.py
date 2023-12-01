from src.data.configuration import ConfigurationManager
from src.data.components.data_preprocessing import DataPreprocessing

class DataPreprocessingPipeline:
    def __init__(self):
        pass

    def main(self):           
        config = ConfigurationManager()
        data_preprocessing_config = config.get_data_preprocessing_config()
        data_preprocessing = DataPreprocessing(config=data_preprocessing_config)
        data_preprocessing.convert()