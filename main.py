import sys
from src.config import ConfigurationManager
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger.logger import logger
from src.exception.exception import PhraserException

STAGE_NAME_1 = "Data Ingestion Stage"
STAGE_NAME_2 = "Data Transformation Stage"
STAGE_NAME_3 = "Model Trainer Stage"

if __name__ == "__main__":
    try:
        # Stage 1: Data Ingestion
        logger.info(f">>>>>> Stage {STAGE_NAME_1} started <<<<<<")
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_data()
        logger.info(f">>>>>> Stage {STAGE_NAME_1} completed successfully <<<<<<")

        # Stage 2: Data Transformation
        logger.info(f">>>>>> Stage {STAGE_NAME_2} started <<<<<<")
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.convert_to_features()
        logger.info(f">>>>>> Stage {STAGE_NAME_2} completed successfully <<<<<<")

        # Stage 3: Model Trainer
        logger.info(f">>>>>> Stage {STAGE_NAME_3} started <<<<<<")
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
        model_trainer.train()
        logger.info(f">>>>>> Stage {STAGE_NAME_3} completed successfully <<<<<<\n")
    except Exception as e:
        logger.error("An error occurred during pipeline execution")
        raise PhraserException(e, sys)