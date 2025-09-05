from phishdetect.baseline import run_tfidf_plus_url

if __name__ == "__main__":
    print("\n=== Baseline: TF–IDF + URL features + Logistic Regression ===")
    print(run_tfidf_plus_url("data/processed/sample.csv"))

