import os
import pandas as pd
from src.logging import logger
from datasets import Dataset, DatasetDict
from src.data.entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def __split_data(self, dataset: Dataset):
        split_dataset = dataset.train_test_split(train_size=self.config.train_data_split, shuffle=True, seed=42)
        test_dataset = split_dataset['test'].train_test_split(train_size=self.config.test_data_split, shuffle=True, seed=42)

        dataset = DatasetDict({
            'train' : split_dataset['train'],
            'test' : test_dataset['train'],
            'eval' : test_dataset['test'],
        })

        return dataset
    
    def __transform_data(self, dataset: Dataset):
        """Transforms the data to the format required by the model"""
        return dataset

    def convert(self):
        finetuning_dataset_df = pd.read_csv(self.config.finetuning_data_path)
        finetuning_dataset = Dataset.from_pandas(finetuning_dataset_df)
        logger.info("Successfully loaded the finetuning data")

        transformed_dataset = self.__transform_data(finetuning_dataset)
        logger.info("Successfully transformed the finetuning data")

        splitted_dataset = self.__split_data(transformed_dataset)
        train_dataset = splitted_dataset['train']
        test_dataset = splitted_dataset['test']
        eval_dataset = splitted_dataset['eval']
        logger.info("Successfully splitted the data")

        train_dataset_df = train_dataset.to_pandas()
        test_dataset_df = test_dataset.to_pandas()
        eval_dataset_df = eval_dataset.to_pandas()
        train_dataset_df.to_csv(os.path.join(self.config.root_dir,"train_dataset.csv"), index=False)
        test_dataset_df.to_csv(os.path.join(self.config.root_dir,"test_dataset.csv"), index=False)
        eval_dataset_df.to_csv(os.path.join(self.config.root_dir,"eval_dataset.csv"), index=False)
        logger.info("Successfully saved the transformed data")