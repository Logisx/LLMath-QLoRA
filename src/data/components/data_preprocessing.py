import os
import pandas as pd
from src.logging import logger
from datasets import Dataset
from src.data.entity import DataPreprocessingConfig

class DataPreprocessing:
    def __init__(self, config: DataPreprocessingConfig):
        self.config = config
        
    def __form_finetuning_dataset(self, dataset_dict: dict, question_key: str, answer_key: str) -> Dataset:
        instruction_template = """{question}"""

        prompt_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

        ### Instruction:

        {instruction}

        ### Response:\n"""

        num_samples = len(dataset_dict[question_key])
        finetuning_dataset_list = []
        for i in range(num_samples):
            question = dataset_dict[question_key][i]
            instruction = instruction_template.format(question=question)
            prompt = prompt_template.format(instruction=instruction)
            response = dataset_dict[answer_key][i] + "\n### End"
            text = prompt + response
            finetuning_dataset_list.append({"instruction": instruction, "response": response, "text": text})

        finetuning_dataset = Dataset.from_list(finetuning_dataset_list)

        return finetuning_dataset
    

    def convert(self):
        instruction_dataset_df = pd.read_csv(self.config.raw_data_path)
        instruction_dataset_dict = instruction_dataset_df.to_dict()
        logger.info("Successfully loaded the raw dataset")

        finetuning_dataset = self.__form_finetuning_dataset(instruction_dataset_dict, question_key = self.config.question_key, answer_key = self.config.answer_key)
        finetuning_dataset_df = finetuning_dataset.to_pandas()
        finetuning_dataset_df.to_csv(os.path.join(self.config.root_dir,"finetuning_dataset.csv"), index=False)
        logger.info("Successfully saved the finetuning dataset")