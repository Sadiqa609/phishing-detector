# 1 Phishing Email Detection System

This repository contains the implementation of a **Machine Learning-based Phishing Email Detection System**, developed as part of a Master's Dissertation project.  

The project combines **Natural Language Processing (NLP)** and **URL-based features** with classical machine learning classifiers to detect phishing emails. It also provides an **explainable web-based interface** built with [Streamlit](https://streamlit.io).

# 2 Features

- Hybrid **TF–IDF + URL** feature extraction.
- Trains and evaluates four ML models:
  - Logistic Regression
  - Support Vector Machine
  - Naïve Bayes
  - Random Forest
- Generates detailed **reports**:
  - Metrics (accuracy, precision, recall, F1, ROC–AUC)
  - Confusion matrix
  - ROC curves
- Provides a **Streamlit UI**:
  - Single email classification
  - Batch email classification (CSV upload)
  - Downloadable prediction results
  - Explainability panel (top phishing & ham n-grams)

---

## 3 Project Structure

| Folder/File                | Description                                      |
|-----------------------------|--------------------------------------------------|
| `app.py`                   | Streamlit application for phishing detection     |
| `train_ui_model.py`        | Trains Logistic Regression model for UI          |
| `train_best_model.py`      | Trains multiple classifiers & saves best model   |
| `evaluate_and_report.py`   | Generates metrics, confusion matrix, ROC curves  |
| `src/phishdetect/`         | Core modules (predictor, URL featurizer)         |
| `models/`                  | Trained ML models (`.joblib`)                    |
| `reports/`                 | Evaluation outputs (JSON, CSV, plots)            |
| `data/processed/`          | Preprocessed dataset (`emails_merged.csv`)       |
| `requirements.txt`         | Dependencies                                     |
| `README.md`                | Project documentation (this file)                |

---

## 4 Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/phishing-detector.git
cd phishing-detector
Create a virtual environment and install dependencies:

bash
Copy code
python -m venv .venv
source .venv/bin/activate   # On Mac/Linux
.venv\Scripts\activate      # On Windows

pip install -r requirements.txt
Training and Evaluation
Train and save the best model:

bash
Copy code
python train_best_model.py
Generate evaluation reports:

bash
Copy code
python evaluate_and_report.py
Outputs will be saved under reports/:

metrics.json

results.csv

confusion_matrix.png

roc_curves.png

Running the Application
Launch the Streamlit app:

bash
Copy code
streamlit run app.py
The app provides:

Single email classification (paste text → get prediction).

Batch classification (upload CSV → download predictions).

Explainability (view top phishing/ham n-grams).

Dataset
The dataset combines:

Phishing emails (Nazario corpus, APWG feeds).

Ham emails (Enron corpus, other legitimate sources).

Stored as: data/processed/emails_merged.csv

Columns:

body → email text

label → 1 = phishing, 0 = ham

Results
Sample performance (actual values in reports/results.csv):

Model	Accuracy	Precision	Recall	F1-score	ROC–AUC
Logistic Regression	0.94	0.93	0.95	0.94	0.96
SVM	0.93	0.92	0.94	0.93	0.95
Naïve Bayes	0.89	0.90	0.87	0.88	0.91
Random Forest	0.96	0.95	0.97	0.96	0.98

Home UI

Batch classification results

Confusion Matrix

ROC Curves

License
This project is released under the MIT License.
Feel free to use, modify, and distribute with attribution.

Acknowledgements
Anti-Phishing Working Group (APWG)

Enron Email Dataset

Nazario Phishing Corpus

Scikit-learn, Pandas, Streamlit