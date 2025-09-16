from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from phishdetect.features.urls import URLFeaturizer

@dataclass
class TextUrlPipeline:
    vec: TfidfVectorizer
    url: URLFeaturizer
    clf: LogisticRegression

    def _featurize(self, texts):
        X_text = self.vec.transform(texts)
        X_url = self.url.transform(list(texts))
        return hstack([X_text, csr_matrix(X_url)])

    def predict_proba_one(self, text: str) -> float:
        X = self._featurize([text])
        return float(self.clf.predict_proba(X)[0, 1])

    def predict_one(self, text: str, threshold: float = 0.5) -> int:
        return int(self.predict_proba_one(text) >= threshold)

def train_and_save(
    csv_path: str = "data/processed/emails_merged.csv",
    out_path: str = "models/texturl_logreg.joblib",
    seed: int = 42
) -> Tuple[str, int]:
    df = pd.read_csv(csv_path)
    assert {"body", "label"}.issubset(df.columns), "CSV must have body,label"
    y = df["label"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        df["body"], y, test_size=0.2, random_state=seed, stratify=y
    )

    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        strip_accents="unicode",
        lowercase=True,
    )
    Xtr_text = vec.fit_transform(X_train)
    url = URLFeaturizer()
    Xtr_url = url.transform(X_train.tolist())
    Xtr = hstack([Xtr_text, csr_matrix(Xtr_url)])

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Xtr, y_train)

    # quick report
    Xte = hstack([vec.transform(X_test), csr_matrix(url.transform(X_test.tolist()))])
    report = classification_report(y_test, clf.predict(Xte))
    print(report)

    pipe = TextUrlPipeline(vec=vec, url=url, clf=clf)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_path)
    return out_path, len(df)

def load_pipeline(model_path: str = "models/texturl_logreg.joblib") -> TextUrlPipeline:
    return joblib.load(model_path)

