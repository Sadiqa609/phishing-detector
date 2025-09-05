from __future__ import annotations
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from scipy.sparse import csr_matrix, hstack

from phishdetect.features.urls import URLFeaturizer

def run_tfidf_plus_url(data_path: str | Path = "data/processed/sample.csv", seed: int = 42) -> str:
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path.resolve()}")

    df = pd.read_csv(data_path)
    if not {"body", "label"}.issubset(df.columns):
        raise ValueError("CSV must have columns: body,label")
    df["label"] = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        df["body"], df["label"], test_size=0.4, random_state=seed, stratify=df["label"]
    )

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, strip_accents="unicode", lowercase=True)
    Xtr_text = vec.fit_transform(X_train)
    Xte_text = vec.transform(X_test)

    url_feat = URLFeaturizer()
    Utr = url_feat.transform(X_train.tolist())
    Ute = url_feat.transform(X_test.tolist())

    Xtr = hstack([Xtr_text, csr_matrix(Utr)])
    Xte = hstack([Xte_text, csr_matrix(Ute)])

    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, y_train)
    y_pred = clf.predict(Xte)

    report = classification_report(y_test, y_pred, digits=4)
    return report

