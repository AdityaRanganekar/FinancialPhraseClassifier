import os
import sys
from datasets import load_dataset
from src.logger.logger import logger
from src.entity import DataIngestionConfig
from src.exception.exception import PhraserException

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_data(self):
        try:
            if not os.path.exists(self.config.local_data_file):
                logger.info(f"Downloading dataset {self.config.dataset_name} from Hugging Face Hub...")
                dataset = load_dataset(self.config.dataset_name)
                dataset.save_to_disk(self.config.local_data_file)
                logger.info(f"Dataset successfully saved at {self.config.local_data_file}")
            else:
                logger.info(f"Dataset already exists at {self.config.local_data_file}. Skipping download.")
        except Exception as e:
            raise PhraserException(e, sys)