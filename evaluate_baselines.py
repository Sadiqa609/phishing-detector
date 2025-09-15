from __future__ import annotations
from pathlib import Path
import json

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy.sparse import hstack, csr_matrix
import matplotlib.pyplot as plt

from phishdetect.features.urls import URLFeaturizer

DATA_PATH = Path("data/processed/emails_merged.csv")
REPORTS_DIR = Path("reports")
SEED = 42

def make_features(train_text: pd.Series, test_text: pd.Series):
    vec = TfidfVectorizer(
        ngram_range=(1,2),
        min_df=2,
        strip_accents="unicode",
        lowercase=True,
    )
    Xtr_text = vec.fit_transform(train_text)
    Xte_text = vec.transform(test_text)

    urlf = URLFeaturizer()
    Utr = urlf.transform(train_text.tolist())
    Ute = urlf.transform(test_text.tolist())

    Xtr = hstack([Xtr_text, csr_matrix(Utr)])
    Xte = hstack([Xte_text, csr_matrix(Ute)])
    return vec, urlf, Xtr, Xte

def train_logreg(Xtr, ytr):
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Xtr, ytr)
    return clf

def train_linear_svm(Xtr, ytr):
    base = LinearSVC(class_weight="balanced")
    # Calibrate to get probabilities & better thresholding for metrics like ROC later
    clf = CalibratedClassifierCV(base, cv=3)
    clf.fit(Xtr, ytr)
    return clf

def plot_confusion(cm: np.ndarray, labels: list[str], out_path: Path, title="Confusion Matrix"):
    fig = plt.figure(figsize=(4,4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    # write numbers
    thresh = cm.max() / 2.0 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

def main():
    if not DATA_PATH.exists():
        raise SystemExit(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    assert {"body","label"}.issubset(df.columns), "CSV must have body,label"
    df["label"] = df["label"].astype(int)

    # train/val/test split: 80/10/10 (stratified)
    X_temp, X_test, y_temp, y_test = train_test_split(
        df["body"], df["label"], test_size=0.10, random_state=SEED, stratify=df["label"]
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1111, random_state=SEED, stratify=y_temp
    )  # 0.1111 * 0.9 ≈ 0.1 overall val

    vec, urlf, Xtr, Xva = make_features(X_train, X_val)
    _,   _,    _,  Xte = make_features(pd.concat([X_train, X_val]), X_test)  # fit on train+val, transform test later

    # Refit vectorizer/url on train+val to avoid val leakage into test only via hyperparams
    vec2 = TfidfVectorizer(ngram_range=(1,2), min_df=2, strip_accents="unicode", lowercase=True)
    Xtrval_text = vec2.fit_transform(pd.concat([X_train, X_val]))
    urlf2 = URLFeaturizer()
    Utrval = urlf2.transform(pd.concat([X_train, X_val]).tolist())
    Xtrval = hstack([Xtrval_text, csr_matrix(Utrval)])

    Xte_text = vec2.transform(X_test)
    Ute = urlf2.transform(X_test.tolist())
    Xte = hstack([Xte_text, csr_matrix(Ute)])

    # Train two models
    logreg = train_logreg(Xtrval, pd.concat([y_train, y_val]).values)
    linsvm = train_linear_svm(Xtrval, pd.concat([y_train, y_val]).values)

    # Evaluate
    results = {}
    for name, model in [("logreg", logreg), ("linear_svm", linsvm)]:
        y_pred = model.predict(Xte)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, digits=4)
        results[name] = {"accuracy": acc, "report": report}

    # Save metrics
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(results, indent=2))

    # Pick better model for confusion matrix
    best_name = max(results.keys(), key=lambda k: results[k]["report"]["weighted avg"]["f1-score"])
    best_model = logreg if best_name == "logreg" else linsvm

    cm = confusion_matrix(y_test, best_model.predict(Xte), labels=[0,1])
    plot_confusion(cm, ["ham(0)","phish(1)"], REPORTS_DIR / "confusion_matrix.png",
                   title=f"Confusion Matrix ({best_name})")

    print("\n=== Evaluation complete ===")
    for k, v in results.items():
        wa = v["report"]["weighted avg"]
        print(f"{k:10s}  acc={v['accuracy']:.4f}  f1_w={wa['f1-score']:.4f}  precision_w={wa['precision']:.4f}  recall_w={wa['recall']:.4f}")
    print("Saved:", REPORTS_DIR / "metrics.json")
    print("Saved:", REPORTS_DIR / "confusion_matrix.png")

if __name__ == "__main__":
    main()

