from pathlib import Path 
from dataclasses import dataclass

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    hf_dataset_name: str
    hf_dataset_split: str
    local_data_file: Path
    unzip_dir: Path   

@dataclass(frozen=True)
class DataPreprocessingConfig:
    root_dir: Path
    raw_data_path: Path
    question_key: str
    answer_key: str