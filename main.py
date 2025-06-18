import streamlit as st
import pandas as pd
from model_utils import load_trained_model, preprocess_input
from config import MODEL_PATH
from train_model import train_model_main  # For retraining feature

# MUST be the first Streamlit command
st.set_page_config(page_title="🎬 Sentiment Analyzer", layout="centered")

# Cache model loading


@st.cache_resource
def get_model():
    return load_trained_model(MODEL_PATH)


# Reload model after retraining
if st.sidebar.button("🔄 Reload Model"):
    st.cache_resource.clear()
    st.success("🔄 Model reloaded successfully!")


def predict_sentiment(review: str):
    model = get_model()
    padded = preprocess_input(review)
    prob = model.predict(padded, verbose=0)[0][0]
    sentiment = "Positive 😊" if prob > 0.5 else "Negative 😞"
    confidence = round(prob if prob > 0.5 else 1 - prob, 2)
    return sentiment, confidence


def batch_predict(reviews):
    results = []
    for review in reviews:
        sentiment, confidence = predict_sentiment(review)
        results.append(
            {"Review": review, "Sentiment": sentiment, "Confidence": confidence})
    return pd.DataFrame(results)


# ------------------ UI Layout ------------------

st.title("🎬 Movie Review Sentiment Analyzer")

# Sidebar Navigation
st.sidebar.title("📊 Navigation")

# dark_mode = st.sidebar.checkbox("🌙 Dark Mode (requires restart)")

# if dark_mode:
#     st.sidebar.info(
#         "To apply dark mode:\n1. Create `.streamlit/config.toml`\n2. Add:\n\n[theme]\nbase=\"dark\"\n\nThen rerun the app."
#     )

# Retrain Model Option
if st.sidebar.checkbox("⚠️ Enable Model Retraining"):
    if st.sidebar.button("🧠 Retrain Sentiment Model"):
        with st.spinner("Training the model... This may take a few minutes."):
            train_model_main()
            st.success("✅ Model retrained and saved successfully!")

mode = st.sidebar.radio(
    "Choose a mode:", ["Single Review", "Batch Processing"])

# ------------------ Single Review Mode ------------------

if mode == "Single Review":
    st.subheader("✏️ Analyze a Single Review")
    with st.form("single_review_form"):
        review_input = st.text_area("Enter your movie review:", height=150)
        submitted = st.form_submit_button("🔍 Analyze Sentiment")

        if submitted:
            if not review_input.strip():
                st.warning("Please enter a valid review.")
            else:
                sentiment, confidence = predict_sentiment(review_input)
                st.success(
                    f"**Sentiment:** {sentiment}  \n**Confidence:** {confidence * 100:.1f}%")

# ------------------ Batch Processing Mode ------------------

elif mode == "Batch Processing":
    st.subheader("📄 Analyze Multiple Reviews")
    uploaded_file = st.file_uploader(
        "Upload a `.txt` (1 review per line) or `.csv` file with a 'review' column:",
        type=['txt', 'csv']
    )

    if uploaded_file:
        st.success(f"✅ Successfully uploaded **{uploaded_file.name}**")

        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
                if "review" not in df.columns:
                    st.error("CSV file must contain a 'review' column.")
                else:
                    result_df = batch_predict(
                        df['review'].astype(str).tolist())
                    st.dataframe(result_df)
                    st.download_button("📥 Download Results", result_df.to_csv(index=False),
                                       file_name="sentiment_predictions.csv")
            else:
                lines = [line.decode('utf-8').strip()
                         for line in uploaded_file.readlines()]
                non_empty_reviews = [line for line in lines if line]
                result_df = batch_predict(non_empty_reviews)
                st.dataframe(result_df)
                st.download_button("📥 Download Results", result_df.to_csv(index=False),
                                   file_name="sentiment_predictions.csv")
        except Exception as e:
            st.error(f"Failed to process file: {e}")

# ------------------ Footer ------------------

st.markdown("---")
st.markdown("Made with ❤️ by DiaMaBo")
