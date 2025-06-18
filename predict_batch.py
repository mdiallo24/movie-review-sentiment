import json
from model_utils import load_trained_model, predict_batch_sentiment


if __name__ == "__main__":
    # Load the trained model and prepare for predictions
    model = load_trained_model()

    # Example input reviews
    review_list = [
        "The movie was amazing and full of surprises!",
        "It was a waste of time and very boring.",
        "I enjoyed the acting but the plot was predictable.",
        "Terrible effects and awful script.",
        "Absolutely brilliant. A must-watch!"
    ]

    # Get predictions
    results = predict_batch_sentiment(model, review_list)

    # Ensure results are JSON serializable
    if isinstance(results, dict):
        # Convert any NumPy arrays or non-serializable objects to lists
        results = {key: value.tolist() if hasattr(value, 'tolist') else value for key, value in results.items()}
    elif isinstance(results, list):
        # Convert list elements if they are NumPy arrays
        results = [item.tolist() if hasattr(item, 'tolist') else item for item in results]

    # Print results as JSON
    print(json.dumps(results, indent=2))
