import sys
from src.config import ConfigurationManager
from src.components.data_ingestion import DataIngestion
from src.logger.logger import logger
from src.exception.exception import PhraserException

STAGE_NAME = "Data Ingestion Stage"

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")

        config = ConfigurationManager()

        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_data()
        
        logger.info(f">>>>>> Stage {STAGE_NAME} completed successfully <<<<<<\n")
        
    except Exception as e:
        logger.error(f"Error occurred during {STAGE_NAME}")
        raise PhraserException(e, sys)