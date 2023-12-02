from src.data.configuration import ConfigurationManager
from src.data.components.data_ingestion import DataIngestion
from src.logging import logger

class DataIngestionPipeline:
    def __init__(self):
        pass

    def main(self):           
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()

if __name__ == "__main__":
    try:
        logger.info('>>>>> Data Ingestion started <<<<<')

        data_ingestion_pipeline = DataIngestionPipeline()
        data_ingestion_pipeline.main()

        logger.info('>>>>> Data Ingestion completed <<<<<')

    except Exception as e:
        logger.exception(e)
        raise e
    