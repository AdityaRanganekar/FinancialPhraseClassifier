import os
from transformers import AutoTokenizer
from datasets import load_from_disk
from src.logger.logger import logger
from src.entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    def convert_to_features(self):
        logger.info(f"Loading ingested dataset from: {self.config.data_path}")
        dataset = load_from_disk(self.config.data_path)

        logger.info(f"Tokenizing text using tokenizer: {self.config.tokenizer_name}")
        tokenized_dataset = dataset.map(
            lambda batch: self.tokenizer(batch["text"], truncation=True),
            batched=True
        )

        output_path = os.path.join(self.config.root_dir, "finbert_dataset")
        tokenized_dataset.save_to_disk(output_path)
        logger.info(f"Tokenized dataset successfully saved at: {output_path}")