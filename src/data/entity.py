from pathlib import Path 
from dataclasses import dataclass

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    hf_dataset_name: str
    hf_dataset_split: str
    local_data_file: Path
    unzip_dir: Path   