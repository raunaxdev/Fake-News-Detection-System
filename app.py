import pickle
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.pkl"
VECTORIZER_PATH = ROOT / "vectorizer.pkl"


@st.cache_resource
def load_artifacts():
    missing_files = [
        path.name for path in (MODEL_PATH, VECTORIZER_PATH) if not path.is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"Missing model artifact(s): {', '.join(missing_files)}. "
            "Run `python train_model.py` first."
        )
    with MODEL_PATH.open("rb") as file:
        model = pickle.load(file)
    with VECTORIZER_PATH.open("rb") as file:
        vectorizer = pickle.load(file)
    return model, vectorizer


st.set_page_config(page_title="Smart Fake News Detection System", page_icon="📰")
st.title("Smart Fake News Detection System")
st.write(
    "Paste a headline or article below to see how the demonstration text classifier "
    "labels it."
)
st.warning(
    "This educational demo uses synthetic training data. Its prediction is not "
    "evidence that a story is true or false; verify claims with reliable sources."
)

news_text = st.text_area("News text", height=220, placeholder="Paste news text here...")

if st.button("Predict / Verify News", type="primary"):
    if not news_text.strip():
        st.warning("Enter some news text before requesting a prediction.")
    else:
        try:
            model, vectorizer = load_artifacts()
        except (FileNotFoundError, OSError, pickle.UnpicklingError) as error:
            st.error(f"Could not load the trained model: {error}")
        else:
            prediction = int(model.predict(vectorizer.transform([news_text]))[0])
            if prediction == 1:
                st.error("Prediction: FAKE")
            else:
                st.success("Prediction: REAL")
