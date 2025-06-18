import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential

# Ensure MODEL_PATH is defined in config.py
from config import MAX_FEATURES, MAXLEN, MODEL_PATH

# --------------------------- Dataset Utilities ---------------------------


def load_and_prepare_data(max_features=MAX_FEATURES, maxlen=MAXLEN):
    """
    Load and preprocess the IMDB dataset.
    Pads all sequences to uniform length.
    """
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

    return x_train, y_train, x_test, y_test


# ---------------------------- Word Index ----------------------------

# Cache the word index and reverse lookup
word_index = imdb.get_word_index()
reverse_word_index = {value: key for (key, value) in word_index.items()}


# ---------------------------- Model Utilities ----------------------------

def load_trained_model(model_path=MODEL_PATH):
    """
    Load and compile a trained Keras model from disk.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")

    print(f"Loading model from '{model_path}'...")
    model = load_model(model_path)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    print("✅ Model loaded and compiled successfully.")
    return model


# ---------------------------- NLP Utilities ----------------------------

def decode_review(encoded_review):
    """
    Convert an encoded IMDB review into human-readable text.
    """
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])


def preprocess_input(review):
    """
    Convert raw review string to a padded sequence for model prediction.
    """
    words = review.lower().split()
    encoded = [min(word_index.get(word, 2) + 3, MAX_FEATURES - 1)
               for word in words]
    padded = sequence.pad_sequences([encoded], maxlen=MAXLEN)
    return padded

# ---------------------------- Prediction Utilities ----------------------------


def predict_sentiment(model, review: str):
    """
    Predict the sentiment of a single review.
    Returns the sentiment label and confidence score.
    """
    padded = preprocess_input(review)  # Preprocess the input review
    # Ensure prob is a Python float
    prob = float(model.predict(padded, verbose=0)[0][0])
    sentiment = "Positive 😊" if prob > 0.5 else "Negative 😞"
    confidence = round(prob if prob > 0.5 else 1 - prob, 2)
    return sentiment, confidence


def predict_batch_sentiment(model, reviews):
    """
    Predict sentiments for a batch of reviews.
    Returns a list of dictionaries with review, sentiment, and confidence.
    """
    predictions = []
    for review in reviews:
        sentiment, confidence = predict_sentiment(model, review)
        predictions.append({
            "review": review,
            "sentiment": sentiment,
            "confidence": confidence
        })
    return predictions


def predict_batch_sentiment2(model, reviews):
    """
    Predict sentiments for a batch of reviews.
    Returns a list of dictionaries with review and sentiment.
    """
    predictions = []
    for review in reviews:
        padded = preprocess_input(review)
        pred = model.predict(padded, verbose=0)[0][0]
        label = "Positive" if pred > 0.5 else "Negative"
        predictions.append({"review": review, "sentiment": label})
    return predictions


def load_and_predict(model_path, reviews):
    """
    Load the model and predict sentiments for a list of reviews.
    """
    model = load_trained_model(model_path)
    if not isinstance(model, Sequential):
        raise ValueError("Loaded model is not a valid Keras Sequential model.")

    return predict_batch_sentiment(model, reviews)
