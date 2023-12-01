from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    data_path: Path
    base_model: str
    training_name: str
    upload_from_hf: bool
    hf_model_name: str

@dataclass(frozen=True)
class LoraParameters:
    r: int
    target_modules: list
    lora_alpha: float
    lora_dropout: float
    bias: str
    task_type: str

@dataclass(frozen=True)
class BitsAndBytesParameters:
    load_in_4bit: bool
    bnb_4bit_quant_type: str
    bnb_4bit_use_double_quant: bool

@dataclass(frozen=True)
class TrainingArgumentsParameters:
    output_dir: str
    evaluation_strategy: str
    save_strategy: str
    num_train_epochs: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    optim: str
    learning_rate: float
    fp16: bool
    max_grad_norm: float
    warmup_ratio: float
    group_by_length: bool
    lr_scheduler_type: str

@dataclass(frozen=True)
class ModelPredictionConfig:
    data_path: Path
    base_model: str
    adapters_path: Path

@dataclass(frozen=True)
class ModelPredictionParameters:
    length_penalty: float
    num_beams: int
    max_length: int