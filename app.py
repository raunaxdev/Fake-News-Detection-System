import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "dataset" / "news.csv"

st.set_page_config(page_title="Smart Fake News Detection System", page_icon="📰")

st.title("Smart Fake News Detection System")
st.write(
    "Paste a headline or article below to see how the demonstration text classifier labels it."
)
st.warning(
    "This educational demo uses synthetic training data. Its prediction is not "
    "evidence that a story is true or false; verify claims with reliable sources."
)

@st.cache_resource
def load_and_train_model():
    if not DATASET_PATH.is_file():
        raise FileNotFoundError("Dataset file 'dataset/news.csv' not found.")
    
    news = pd.read_csv(DATASET_PATH)
    content = (news["title"].astype(str) + " " + news["text"].astype(str)).str.strip()
    
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    features = vectorizer.fit_transform(content)
    labels = news["label"].astype(int)
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(features, labels)
    
    return model, vectorizer

news_text = st.text_area("News text", height=180, placeholder="Paste news text here...")

if st.button("Predict / Verify News", type="primary"):
    if not news_text.strip():
        st.warning("Enter some news text before requesting a prediction.")
    else:
        try:
            model, vectorizer = load_and_train_model()
            prediction = int(model.predict(vectorizer.transform([news_text]))[0])
            if prediction == 1:
                st.error("🚨 Warning: This news appears to be FAKE!")
            else:
                st.success("✅ Clean: This news appears to be REAL!")
        except Exception as error:
            st.error(f"Error executing prediction: {error}")
