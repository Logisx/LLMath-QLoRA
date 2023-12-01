import torch 
import os
import locale
import math
import mlflow
import pandas as pd
from trl import SFTTrainer
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel, LlamaForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig

from src.logging import logger
from src.models.entity import (ModelTrainerConfig,
                                LoraParameters,
                                BitsAndBytesParameters,
                                TrainingArgumentsParameters)


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, lora_parameters: LoraParameters, bits_and_bytes_parameters: BitsAndBytesParameters, training_arguments: TrainingArgumentsParameters):
        self.model_trainer_config = model_trainer_config
        self.lora_parameters = lora_parameters
        self.bits_and_bytes_parameters = bits_and_bytes_parameters
        self.training_arguments = training_arguments


    def __load_data(self):
        train_dataset = pd.read_csv(os.path.join(self.model_trainer_config.data_path, "train_dataset.csv"))
        eval_dataset = pd.read_csv(os.path.join(self.model_trainer_config.data_path, "eval_dataset.csv"))

        train_dataset = Dataset.from_pandas(train_dataset)
        eval_dataset = Dataset.from_pandas(eval_dataset)

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        logger.info("Data loaded")


    def __initialize_tokenizer(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_trainer_config.base_model)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        logger.info("Tokenizer initialized")


    def __initialize_lora(self):
        self.lora_cofig = LoraConfig(
            r = self.lora_parameters.r,
            target_modules = self.lora_parameters.target_modules,
            lora_alpha = self.lora_parameters.lora_alpha,
            lora_dropout = self.lora_parameters.lora_dropout,
            bias = self.lora_parameters.bias,
            task_type = self.lora_parameters.task_type
        )
        logger.info("Lora initialized")


    def __initialize_bits_and_bytes(self):
        self.nf4_config = BitsAndBytesConfig(
            load_in_4bit = self.bits_and_bytes_parameters.load_in_4bit,
            bnb_4bit_quant_type = self.bits_and_bytes_parameters.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant = self.bits_and_bytes_parameters.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype = torch.bfloat16
        )
        logger.info("Bits and bytes initialized")
    

    def __initialize_training_arguments(self):
        self.training_args = TrainingArguments(
            output_dir = self.training_arguments.output_dir,
            evaluation_strategy = self.training_arguments.evaluation_strategy,
            save_strategy = self.training_arguments.save_strategy,
            num_train_epochs = self.training_arguments.num_train_epochs,
            per_device_train_batch_size = self.training_arguments.per_device_train_batch_size,
            gradient_accumulation_steps = self.training_arguments.gradient_accumulation_steps,
            optim = self.training_arguments.optim,
            learning_rate = self.training_arguments.learning_rate,
            fp16 = self.training_arguments.fp16,
            max_grad_norm = self.training_arguments.max_grad_norm,
            warmup_ratio = self.training_arguments.warmup_ratio,
            group_by_length = self.training_arguments.group_by_length,
            lr_scheduler_type = self.training_arguments.lr_scheduler_type
        )
        logger.info("Training arguments initialized")


    def __create_model(self):
        self.model = LlamaForCausalLM.from_pretrained(
            self.model_trainer_config.base_model, device_map='auto', quantization_config=self.nf4_config,
        )
        self.model = get_peft_model(self.model, self.lora_config)
        #self.model.print_trainable_parameters()
        logger.info("Model created")

    def __evaluate(self, trainer):
        evaluation_results = trainer.evaluate()
        logger.info(f"Perplexity: {math.exp(evaluation_results['eval_loss']):.2f}")

    def __save_model(self):
        self.model.save_pretrained(os.path.join(self.config.root_dir, f"{self.model_trainer_config.base_model}-math-model"))
        logger.info("Model saved")

    def __save_tokenizer(self):
        self.tokenizer.save_pretrained(os.path.join(self.config.root_dir,"tokenizer"))
        logger.info("Tokenizer saved")


    def train(self):
        if self.model_trainer_config.upload_from_hf:
            logger.info("Uploading model from HuggingFace")
            self.__initialize_tokenizer(self.model_trainer_config.hf_model_name)
            self.__initialize_bits_and_bytes()
            
            self.model = AutoModel.from_pretrained(self.model_trainer_config.hf_model_name, device_map='auto', quantization_config=self.nf4_config)

            self.__save_model()
            self.__save_tokenizer()

            return None

        if torch.cuda.is_available():
            try:
                locale.getpreferredencoding = lambda: "UTF-8"
                
                self.__load_data()
                self.__initialize_tokenizer(self.model_trainer_config.base_model)
                self.__initialize_lora()
                self.__initialize_bits_and_bytes()
                self.__initialize_training_arguments()
                self.__create_model()

                trainer = SFTTrainer(self.model,
                                    train_dataset=self.train_dataset,
                                    eval_dataset=self.eval_dataset,
                                    dataset_text_field="text",
                                    max_seq_length=256,
                                    args=self.training_args,
                                    )
                logger.info("Trainer created")
                
                #Upcast layer norms to float 32 for stability
                for name, module in trainer.model.named_modules():
                    if "norm" in name:
                        module = module.to(torch.float32)
                logger.info("Layer norms upcasted to float32")
                
                logger.info(">>>>>>> Training started <<<<<<<<")
                with mlflow.start_run(run_name=self.model_trainer_config.training_name):
                    trainer.train()
                logger.info(">>>>>>> Training completed <<<<<<<<")

                self.__evaluate(trainer)
                
                self.__save_model()
                self.__save_tokenizer()

            except Exception as e:
                raise e
        else:
            raise Exception("No GPU found")