import os
import mlflow
from urllib.parse import urlparse
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_from_disk
from src.logger.logger import logger
from src.entity import ModelEvaluationConfig
from src.utils.common import save_json

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def evaluate_and_log(self):
        logger.info("Loading dataset and model for evaluation...")
        dataset = load_from_disk(self.config.data_path)
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        model = AutoModelForSequenceClassification.from_pretrained(self.config.model_path)

        training_args = TrainingArguments(
            output_dir=self.config.root_dir,
            per_device_eval_batch_size=16,
            use_cpu=True
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            processing_class=tokenizer,
            eval_dataset=dataset["validation"]
        )

        logger.info("Running evaluation on validation set...")
        predictions = trainer.predict(dataset["validation"])
        preds = np.argmax(predictions.predictions, axis=-1)
        labels = predictions.label_ids

        eval_metrics = {
            "accuracy": accuracy_score(labels, preds),
            "f1_score": f1_score(labels, preds, average="macro")
        }

        save_json(path=self.config.metric_file_name, data=eval_metrics)

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(eval_metrics)

            if tracking_url_type_store != "file":
                mlflow.transformers.log_model(
                    transformers_model={"model": model, "tokenizer": tokenizer},
                    name = "model",
                    task="text-classification",
                    registered_model_name="FinBERT_Sentiment_Model"
                )
            else:
                mlflow.transformers.log_model(
                    transformers_model={"model": model, "tokenizer": tokenizer},
                    name="model"
                )
        logger.info("Metrics and model successfully logged to DagsHub via MLflow.")