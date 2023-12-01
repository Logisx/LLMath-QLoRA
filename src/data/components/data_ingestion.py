import os
import zipfile
import pandas as pd
from pathlib import Path
from src.logging import logger
import urllib.request as request
from datasets import load_dataset
from src.data.entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        
    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            raw_dataset = load_dataset(self.config.hf_dataset_name, split=self.config.hf_dataset_split)
            df = pd.DataFrame(raw_dataset)  
            
            with zipfile.ZipFile(self.config.local_data_file, 'w') as z:
                df.to_csv('raw_dataset.csv', index=False)  # Save DataFrame to CSV file
                z.write('raw_dataset.csv')  # Write CSV file to the zip archive
            
            os.remove('raw_dataset.csv')  # Remove the temporary CSV file after zipping
            logger.info(f"Dataset {self.config.hf_dataset_name} downloaded and archived as data.zip!")
        else:
            logger.info(f"File already exists. File size: {Path(self.config.local_data_file).stat().st_size}") 
    
    
    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
        logger.info(f"Data extracted at {unzip_path}")