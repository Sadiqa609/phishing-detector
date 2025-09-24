from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from scipy.sparse import hstack, csr_matrix

from phishdetect.features.urls import URLFeaturizer

SEED = 42
DATA = Path("data/processed/emails_merged.csv")
REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)

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
        ),
    }

def main():
    df = pd.read_csv(DATA)
    y = df["label"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        df["body"], y, test_size=0.2, stratify=y, random_state=SEED
    )

    vec, urlf, Xtr, Xte = featurize(X_train, X_test)
    models = get_models()

    results = {}
    roc_curves = {}

    for name, model in models.items():
        print(f"Training {name}…")
        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)
        acc = accuracy_score(y_test, y_pred)
        rep = classification_report(y_test, y_pred, output_dict=True, digits=4)

        # Scores for ROC
        try:
            scores = model.predict_proba(Xte)[:, 1]
        except Exception:
            s = model.decision_function(Xte)
            scores = (s - s.min()) / (s.max() - s.min() + 1e-9)

        fpr, tpr, _ = roc_curve(y_test, scores)
        roc_auc = auc(fpr, tpr)

        results[name] = {"accuracy": float(acc), "auc": float(roc_auc), "report": rep}
        roc_curves[name] = (fpr, tpr, roc_auc)

    # 1) metrics.json
    (REPORTS / "metrics.json").write_text(json.dumps(results, indent=2))

    # 2) results.csv
    rows = []
    for name, r in results.items():
        wa = r["report"]["weighted avg"]
        rows.append([name, r["accuracy"], wa["precision"], wa["recall"], wa["f1-score"], r["auc"]])
    pd.DataFrame(rows, columns=["model", "accuracy", "precision", "recall", "f1", "auc"]).to_csv(REPORTS / "results.csv", index=False)

    # 3) confusion_matrix.png (best by weighted F1)
    best = max(results.keys(), key=lambda k: results[k]["report"]["weighted avg"]["f1-score"])
    best_model = get_models()[best].fit(Xtr, y_train)
    cm = confusion_matrix(y_test, best_model.predict(Xte), labels=[0, 1])

    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Ham(0)", "Phish(1)"])
    ax.set_yticklabels(["Ham(0)", "Phish(1)"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(REPORTS / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    # 4) roc_curves.png
    plt.figure(figsize=(6, 5))
    for name, (fpr, tpr, roc_auc) in roc_curves.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", alpha=0.5)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves"); plt.legend(); plt.tight_layout()
    plt.savefig(REPORTS / "roc_curves.png", dpi=160)
    plt.close()

    print("\n=== Done ===")
    print("Saved metrics.json, results.csv, confusion_matrix.png, roc_curves.png in reports/")

if __name__ == "__main__":
    main()
