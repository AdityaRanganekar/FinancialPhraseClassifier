from dataclasses import dataclass
from pathlib import Path

@dataclass()
class DataIngestionConfig:
    root_dir: Path
    dataset_name: str
    local_data_file: Path

@dataclass()
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    tokenizer_name: str

@dataclass()
class ModelTrainerConfig:
    root_dir: Path
    data_path: Path
    model_ckpt: str
    num_train_epochs: int
    learning_rate: float
    per_device_train_batch_size: int
    weight_decay: float

@dataclass()
class ModelEvaluationConfig:
    root_dir: Path
    data_path: Path
    model_path: Path
    tokenizer_path: Path
    metric_file_name: Path
    mlflow_uri: str
    all_params: dict