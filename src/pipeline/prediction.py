import os
from transformers import pipeline

class PredictionPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model_trainer", "finbert-model")
        
        self.classifier = pipeline(
            "text-classification", 
            model=self.model_path, 
            tokenizer=self.model_path
        )

    def predict(self, text: str):
        prediction = self.classifier(text)[0]
        return prediction