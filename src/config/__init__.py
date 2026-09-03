import os
from pathlib import Path
from src.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from src.utils.common import read_yaml
from src.entity import DataIngestionConfig, DataTransformationConfig, ModelTrainerConfig

class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        os.makedirs(self.config.artifacts_root, exist_ok=True)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        os.makedirs(config.root_dir, exist_ok=True)

        return DataIngestionConfig(
            root_dir=Path(config.root_dir),
            dataset_name=config.dataset_name,
            local_data_file=Path(config.local_data_file)
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        os.makedirs(config.root_dir, exist_ok=True)

        return DataTransformationConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            tokenizer_name=config.tokenizer_name
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.TrainingArguments
        
        os.makedirs(config.root_dir, exist_ok=True)

        return ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            model_ckpt=config.model_ckpt,
            num_train_epochs=params.num_train_epochs,
            learning_rate=params.learning_rate,
            per_device_train_batch_size=params.per_device_train_batch_size,
            weight_decay=params.weight_decay
        )