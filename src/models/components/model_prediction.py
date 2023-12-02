import torch
from transformers import pipeline, AutoTokenizer, LlamaForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from src.models.entity import (ModelPredictionConfig,
                                ModelPredictionParameters,
                                BitsAndBytesParameters)
from src.logging import logger

class ModelPrediction:
    def __init__(self, config: ModelPredictionConfig, bits_and_bytes_parameters: BitsAndBytesParameters, params: ModelPredictionParameters):
        self.config = config
        self.bits_and_bytes_parameters = bits_and_bytes_parameters
        self.params = params 

    def __initialize_tokenizer(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        logger.info("Tokenizer initialized")

    def __initialize_bits_and_bytes(self):
        self.nf4_config = BitsAndBytesConfig(
            load_in_4bit = self.bits_and_bytes_parameters.load_in_4bit,
            bnb_4bit_quant_type = self.bits_and_bytes_parameters.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant = self.bits_and_bytes_parameters.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype = torch.bfloat16
        )
        logger.info("Bits and bytes initialized")

    def __initialize_model(self):
        self.model = LlamaForCausalLM.from_pretrained(self.config.base_model, device_map='auto', quantization_config=self.nf4_config)
        self.peft_model = PeftModel.from_pretrained(self.model, self.config.adapters_path)
        logger.info("Model initialized")


    def predict(self, question):
        self.__initialize_tokenizer(self.config.base_model)
        self.__initialize_bits_and_bytes()
        self.__initialize_model()

        gen_kwargs = {"length_penalty": self.params.length_penalty,
                      "num_beams": self.params.max_length,
                      "max_length": self.params.max_length}

        pipe = pipeline("generation", model=self.peft_model, tokenizer=self.tokenizer)
        logger.info("Pipeline initialized")

        logger.info("Generating output...")
        output = pipe(question, **gen_kwargs)[0]["response"]
        logger.info("Output generated: ", output)

        return output