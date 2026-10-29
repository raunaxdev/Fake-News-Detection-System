# Smart Fake News Detection System

A small Streamlit demonstration that uses TF-IDF features and logistic regression
to classify pasted text as `FAKE` or `REAL`.

## Important limitation

The included 600-row CSV is **synthetic and fictional**, created only to make the
demo runnable. The classifier learns wording patterns in that artificial dataset;
it does not verify claims, consult current sources, or establish whether real-world
reporting is true. Its predictions must not be used to make consequential
decisions. Verify information with multiple reliable sources.

## Project files

- `app.py` — Streamlit interface for entering text and viewing a prediction.
- `train_model.py` — trains and evaluates the text classifier and writes its
  artifacts.
- `dataset/news.csv` — synthetic examples with `title`, `text`, and `label`
  columns (`1` = FAKE, `0` = REAL).
- `model.pkl`, `vectorizer.pkl` — generated locally by training; excluded from
  version control.

## Setup and run locally

Use Python 3.10 or later. From the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

The training script reports accuracy, precision, recall, and F1-score on a
stratified 20% holdout split, then creates `model.pkl` and `vectorizer.pkl` in the
project directory.

## Deployment

For Streamlit Community Cloud or another Streamlit host:

1. Push the project source, `dataset/news.csv`, `requirements.txt`, and this
   README to a repository.
2. Set `app.py` as the app entry point.
3. Configure the deployment to run `python train_model.py` before
   `streamlit run app.py`, or train locally and provide the matching
   `model.pkl` and `vectorizer.pkl` artifacts through your deployment process.
4. Confirm the deployment has the same compatible Python and scikit-learn
   versions used to create the pickle files.

Pickle files can execute code when loaded. Only load model artifacts that you
created or otherwise trust. Retrain after changing the dataset or relevant
library versions.
