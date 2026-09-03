import os
from pathlib import Path
from src.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from src.utils.common import read_yaml
from src.entity import DataIngestionConfig

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