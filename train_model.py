# Required installations (uncomment if running in a notebook)
# %pip install numpy
# %pip install tensorflow

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# --------------------- Constants ---------------------
from config import MAX_FEATURES, MAXLEN, BATCH_SIZE, EPOCHS, MODEL_PATH

# ------------------- Data Preparation -------------------


def load_and_prepare_data(max_features=MAX_FEATURES, maxlen=MAXLEN):
    """
    Loads and preprocesses the IMDB dataset.
    Pads sequences to ensure consistent input shape.
    """
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

    return x_train, y_train, x_test, y_test


# -------------------- Model Building --------------------
def build_rnn_model(max_features=MAX_FEATURES, maxlen=MAXLEN):
    """
    Builds and returns a Bidirectional LSTM model for binary classification.
    """
    model = Sequential([
        Embedding(max_features, 128, input_length=maxlen),
        Bidirectional(LSTM(128, activation='tanh')),
        Dense(1, activation='sigmoid')
    ])
    model.summary()
    return model


# -------------------- Model Training --------------------
def train_rnn_model(model, x_train, y_train):
    """
    Compiles and trains the model with early stopping.
    """
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    history = model.fit(
        x_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=[early_stopping]
    )
    return history


# -------------------- Evaluation & Save --------------------
def evaluate_and_save_model(model, x_test, y_test, model_path=MODEL_PATH):
    """
    Evaluates the model on the test set and saves it to disk.
    """
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
    model.save(model_path)


# ------------------------ Main ------------------------
def train_model_main():
    x_train, y_train, x_test, y_test = load_and_prepare_data()
    model = build_rnn_model()
    train_rnn_model(model, x_train, y_train)
    evaluate_and_save_model(model, x_test, y_test)


if __name__ == "__main__":
    train_model_main()
# This code is designed to be run in a Python environment with TensorFlow installed.
# It will train a Bidirectional LSTM model on the IMDB dataset and save the trained model to disk.
# The model can then be loaded for making predictions on new reviews.
