from phishdetect.predictor import train_and_save

if __name__ == "__main__":
    model_path, n = train_and_save("data/processed/emails_merged.csv")
    print(f"Saved model to {model_path} (n={n})")

