import streamlit as st
import pandas as pd
import re
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
    cleaned_text = news_text.strip()
    words = re.findall(r'\b[a-zA-Z]{3,}\b', cleaned_text)
    
    if not cleaned_text:
        st.warning("⚠️ Please enter news text before requesting a prediction.")
    elif len(words) < 3:
        st.warning("⚠️ Please enter a valid news sentence/headline containing meaningful words.")
    else:
        try:
            model, vectorizer = load_and_train_model()
            transformed_input = vectorizer.transform([cleaned_text])
            
            # Check if input contains any recognized vocabulary features
            if transformed_input.nnz == 0:
                st.warning("⚠️ The input text contains unrecognized words. Please provide a clear English news article or headline.")
            else:
                prediction = int(model.predict(transformed_input)[0])
                if prediction == 1:
                    st.error("🚨 Warning: This news appears to be FAKE!")
                else:
                    st.success("✅ Clean: This news appears to be REAL!")
        except Exception as error:
            st.error(f"Error executing prediction: {error}")