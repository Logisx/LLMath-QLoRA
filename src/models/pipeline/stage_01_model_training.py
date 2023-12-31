from src.models.configuration import ConfigurationManager
from src.models.components.model_training import ModelTrainer
from src.logging import logger

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):           
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        lora_parameters = config.get_lora_params()
        bits_and_bytes_parameters = config.get_bits_and_bytes_params()
        training_args = config.get_training_args()
        model_trainer = ModelTrainer(model_trainer_config=model_trainer_config, lora_parameters=lora_parameters, bits_and_bytes_parameters=bits_and_bytes_parameters, training_arguments=training_args)
        model_trainer.train()


if __name__ == "__main__":
    try:
        logger.info('>>>>> Model Training started <<<<<')

        model_training_pipeline = ModelTrainingPipeline()
        model_training_pipeline.main()

        logger.info('>>>>> Model Training completed <<<<<')
        
    except Exception as e:
        logger.exception(e)
        raise e