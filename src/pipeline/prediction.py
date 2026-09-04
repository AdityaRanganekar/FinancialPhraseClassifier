import os
from transformers import pipeline

class PredictionPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model_trainer", "finbert-model")

    def predict(self, text: str):

        classifier = pipeline(
            "text-classification", 
            model=self.model_path, 
            tokenizer=self.model_path
        )

        prediction = classifier(text)[0]
        return prediction


if __name__ == "__main__":
    predictor = PredictionPipeline()
    sample_text = "The company reported a massive 25% surge in Q3 revenue, easily beating all analyst expectations."
    
    result = predictor.predict(sample_text)
    print(f"Input: {sample_text}")
    print(f"Prediction: {result['label']} (Confidence: {result['score']:.4f})")