import os
from pathlib import Path
from src.utils.common import read_yaml
from dotenv import find_dotenv, load_dotenv
from src.models.entity import (ModelTrainerConfig,
                                 LoraParameters,
                                 BitsAndBytesParameters,
                                 TrainingArgumentsParameters,
                                 ModelPredictionConfig,
                                 ModelPredictionParameters)


_ = load_dotenv(find_dotenv()) # read local .env file

MODEL_CONFIG_FILE_PATH = os.environ['MODEL_CONFIG_FILE_PATH']
MODEL_PARAMS_FILE_PATH = os.environ['MODEL_PARAMS_FILE_PATH']


class ConfigurationManager:
    def __init__(self,
                model_config_filepath = MODEL_CONFIG_FILE_PATH,
                model_params_filepath = MODEL_PARAMS_FILE_PATH):

        self.config = read_yaml(Path(model_config_filepath))
        self.params = read_yaml(Path(model_params_filepath))


    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        model_trainer_config = ModelTrainerConfig(
            root_dir = Path(config.root_dir),
            data_path = Path(config.data_path),
            base_model = config.base_model,
            training_name = config.training_name,
            upload_from_hf = config.upload_from_hf,
            hf_model_name = config.hf_model_name
        )

        return model_trainer_config
    

    def get_lora_params(self) -> LoraParameters:
        params = self.params.lora_parameters

        lora_parameters = LoraParameters(
            r = params.r,
            target_modules = params.target_modules,
            lora_alpha = params.lora_alpha,
            lora_dropout = params.lora_dropout,
            bias = params.bias,
            task_type = params.task_type
        )

        return lora_parameters
    

    def get_bits_and_bytes_params(self) -> BitsAndBytesParameters:
        params = self.params.bits_and_bytes_parameters

        bits_and_bytes_parameters = BitsAndBytesParameters(
            load_in_4bit = params.load_in_4bit,
            bnb_4bit_quant_type = params.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant = params.bnb_4bit_use_double_quant
        )

        return bits_and_bytes_parameters
    

    def get_training_args(self) -> TrainingArgumentsParameters:
        params = self.params.training_arguments

        training_args = TrainingArgumentsParameters(
            output_dir = params.output_dir,
            evaluation_strategy = params.evaluation_strategy,
            save_strategy = params.save_strategy,
            num_train_epochs = params.num_train_epochs,
            per_device_train_batch_size = params.per_device_train_batch_size,
            gradient_accumulation_steps = params.gradient_accumulation_steps,
            optim = params.optim,
            learning_rate = params.learning_rate,
            fp16 = params.fp16,
            max_grad_norm = params.max_grad_norm,
            warmup_ratio = params.warmup_ratio,
            group_by_length = params.group_by_length,
            lr_scheduler_type = params.lr_scheduler_type
        )

        return training_args


    def get_model_prediction_config(self) -> ModelPredictionConfig:
        config = self.config.model_prediction

        model_prediction_config = ModelPredictionConfig(
            data_path=config.data_path,
            base_model = config.base_model,
            adapters_path = config.adapters_path
        )

        return model_prediction_config
    

    def get_model_prediction_parameters(self) -> ModelPredictionParameters:
        config = self.params.prediction_parameters

        model_prediction_parameters = ModelPredictionParameters(
            length_penalty=config.length_penalty,
            num_beams = config.num_beams,
            max_length = config.max_length
        )

        return model_prediction_parameters