import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import imdb
from model_utils import load_trained_model, decode_review, predict_sentiment
from config import MAXLEN


if __name__ == "__main__":
    # Load the trained model
    model = load_trained_model()

    # Example review
    review = "It was a waste of time and very boring."

    # Predict sentiment
    sentiment, confidence = predict_sentiment(model, review)
    print(f"Review: {review}")
    print(f"Predicted Sentiment: {sentiment} (Confidence: {confidence:.2f})")

    # Decode a sample review from the IMDB dataset
    (x_train, _), _ = imdb.load_data()
    sample_review = x_train[0]
    print(f"\nDecoded Sample Review:\n{decode_review(sample_review)}")
