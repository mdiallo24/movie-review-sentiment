# Natural Language Processing

## Project Description

This project is a complete workflow for customer churn prediction using machine learning and deep learning techniques. It includes:

- **Data preprocessing and feature engineering** using pandas and scikit-learn.
- **Model training** with TensorFlow/Keras, including label encoding, one-hot encoding, and feature scaling.
- **Model evaluation** and saving of trained models and preprocessing objects.
- **Interactive prediction app** built with Streamlit, allowing users to input customer data and receive real-time churn predictions.
- **Jupyter notebooks** for experiments and reproducibility.
- **Utilities** for loading models, encoders, and data.
- **TensorBoard integration** for monitoring training progress.

The project is organized for easy experimentation, deployment, and extension to other NLP or tabular ML tasks.

---

## How to run the app

```
conda create -p venv python==3.11 -y
conda activate venv/

or

python3 -m venv .venv
. .venv/bin/activate

then

pip install -r requirements.txt
pip install ipykernel
streamlit run main.py

pip install --upgrade pydantic typer transformers

pip install --upgrade --use-feature=fast-deps gradio sentence-transformers
```

# Sentiment Analysis with Deep Learning

This project provides a deep learning-based sentiment analysis tool for movie reviews using the IMDB dataset. It features a trained RNN (or LSTM/Bidirectional LSTM) model for binary sentiment classification (positive/negative) and includes utilities for both single and batch prediction, as well as a command-line interface.

## Features

- **Model Training:** Scripts to train a recurrent neural network on the IMDB dataset.
- **Model Saving/Loading:** Save trained models in Keras (`.keras`) or TensorFlow SavedModel format.
- **Prediction Utilities:** Functions for predicting sentiment for single reviews or batches.
- **Command-Line Interface:** Interactive CLI to choose between single or batch review prediction.
- **Streamlit App (optional):** Easily adaptable for web-based sentiment analysis.

## Project Structure

```
.
├── model_utils.py         # Utilities for loading models and making predictions
├── predict.py             # CLI for single/batch sentiment prediction
├── train.py               # Model training script
├── saved_model/           # Directory for SavedModel format
├── bidirectional_lstm_imdb_model.keras  # Example Keras model file
├── requirements.txt       # Python dependencies
└── README.md
```

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model (if not already trained)

```bash
python train_model.py
```

This will save the trained model in the specified format (e.g., `saved_model/` or `.keras`).

### 3. Predict Sentiment

#### Command-Line Interface

```bash
python predict.py
```

You will be prompted to choose between single review or batch processing.

#### Example Output

```
Choose an option:
1. Predict sentiment for a single review
2. Predict sentiment for a batch of reviews
Enter your choice (1 or 2): 1
Enter a review: The movie was fantastic!
Result:
{
  "review": "The movie was fantastic!",
  "sentiment": "Positive 😊",
  "confidence": 0.95
}
```

## Requirements

- Python 3.8+
- TensorFlow
- NumPy
- (Optional) Streamlit

See `requirements.txt` for the full list.

## License

This project is licensed under the MIT License.

---

**Contributions

```
DiaMaBo Dev
```
