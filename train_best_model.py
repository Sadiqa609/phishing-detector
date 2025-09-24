from pathlib import Path
import json
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import hstack, csr_matrix

from phishdetect.features.urls import URLFeaturizer

SEED = 42
DATA = Path("data/processed/emails_merged.csv")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

def featurize(train_text, test_text):
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=2, strip_accents="unicode")
    Xtr_text = vec.fit_transform(train_text)
    Xte_text = vec.transform(test_text)
    urlf = URLFeaturizer()
    Utr = urlf.transform(train_text.tolist())
    Ute = urlf.transform(test_text.tolist())
    Xtr = hstack([Xtr_text, csr_matrix(Utr)])
    Xte = hstack([Xte_text, csr_matrix(Ute)])
    return vec, urlf, Xtr, Xte

def get_models():
    return {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "svm": CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=3),
        "naive_bayes": MultinomialNB(),
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1
        )
    }

def main():
    df = pd.read_csv(DATA)
    y = df["label"].astype(int)
    X_train, X_valid, y_train, y_valid = train_test_split(
        df["body"], y, test_size=0.2, stratify=y, random_state=SEED
    )

    vec, urlf, Xtr, Xva = featurize(X_train, X_valid)
    models = get_models()

    scores = {}
    for name, model in models.items():
        print(f"Training {name}…")
        model.fit(Xtr, y_train)
        pred = model.predict(Xva)
        acc = accuracy_score(y_valid, pred)
        rep = classification_report(y_valid, pred, output_dict=True, digits=4)
        scores[name] = {"accuracy": acc, "f1": rep["weighted avg"]["f1-score"]}

    best = max(scores.keys(), key=lambda k: scores[k]["f1"])
    print(f"\nBest model: {best} (weighted F1={scores[best]['f1']:.4f})")

    # Refit best model on full data for saving
    vec_all = TfidfVectorizer(ngram_range=(1,2), min_df=2, strip_accents="unicode")
    X_text_all = vec_all.fit_transform(df["body"])
    urlf = URLFeaturizer()
    U_all = urlf.transform(df["body"].tolist())
    X_all = hstack([X_text_all, csr_matrix(U_all)])

    best_model = get_models()[best]
    best_model.fit(X_all, y)

    from joblib import dump
    dump({"vec": vec_all, "urlf": urlf, "clf": best_model}, MODELS_DIR/"texturl_best.joblib")
    (REPORTS_DIR/"best_model.json").write_text(json.dumps({"best": best, "scores": scores}, indent=2))
    print("Wrote model to models/texturl_best.joblib and summary to reports/best_model.json")

if __name__ == "__main__":
    main()
