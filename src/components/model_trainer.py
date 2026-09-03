import os
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    DataCollatorWithPadding
)
from datasets import load_from_disk
from src.logger.logger import logger
from src.entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        logger.info(f"Loading tokenized dataset from {self.config.data_path}")
        dataset = load_from_disk(self.config.data_path)
        
        logger.info(f"Initializing tokenizer and data collator...")
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # Map exact 3-class structure used in Colab
        id2label = {0: "bearish", 1: "bullish", 2: "neutral"}
        label2id = {"bearish": 0, "bullish": 1, "neutral": 2}

        logger.info(f"Loading base model: {self.config.model_ckpt}")
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_ckpt,
            num_labels=3,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )

        logger.info("Setting up memory-optimized TrainingArguments...")
        training_args = TrainingArguments(
            output_dir=self.config.root_dir,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=2,
            weight_decay=self.config.weight_decay,
            max_steps=2,
            logging_steps=1,
            save_strategy="no",
            use_cpu=True
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            processing_class=tokenizer,
            data_collator=data_collator,
            train_dataset=dataset["train"].select(range(10)),
            eval_dataset=dataset["validation"].select(range(5))
        )

        logger.info("Executing local model training check...")
        trainer.train()

        save_path = os.path.join(self.config.root_dir, "finbert-model")
        trainer.save_model(save_path)
        tokenizer.save_pretrained(save_path)
        logger.info(f"Model and tokenizer successfully saved to {save_path}")