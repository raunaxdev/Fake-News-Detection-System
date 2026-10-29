import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "dataset" / "news.csv"
MODEL_PATH = ROOT / "model.pkl"
VECTORIZER_PATH = ROOT / "vectorizer.pkl"


def main():
    news = pd.read_csv(DATASET_PATH)
    required_columns = {"title", "text", "label"}
    missing_columns = required_columns.difference(news.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")
    if news[list(required_columns)].isnull().any().any():
        raise ValueError("Dataset contains missing values in required columns.")
    if not set(news["label"].unique()).issubset({0, 1}) or news["label"].nunique() != 2:
        raise ValueError("The label column must contain both binary labels 0 and 1.")

    content = (news["title"].astype(str) + " " + news["text"].astype(str)).str.strip()
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    features = vectorizer.fit_transform(content)
    labels = news["label"].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, predictions, zero_division=0):.4f}")
    print(f"F1-score:  {f1_score(y_test, predictions, zero_division=0):.4f}")

    with MODEL_PATH.open("wb") as file:
        pickle.dump(model, file)
    with VECTORIZER_PATH.open("wb") as file:
        pickle.dump(vectorizer, file)
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved vectorizer to {VECTORIZER_PATH}")


if __name__ == "__main__":
    main()
