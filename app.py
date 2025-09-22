from __future__ import annotations
from pathlib import Path
import io
import json
import numpy as np
import pandas as pd
import streamlit as st

from phishdetect.predictor import load_pipeline
from phishdetect.features.urls import extract_urls, URLFeaturizer

# -----------------------------------------------------------------------------
# Page setup (SINGLE header)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Phishing Email Detector", page_icon="🛡️", layout="wide")
st.title("🛡️ Phishing Email Detector")

MODEL_PATH = Path("models/texturl_logreg.joblib")
DATA_PATH = Path("data/processed/emails_merged.csv")
REPORTS_DIR = Path("reports")
SUSPICIOUS_KWS = ("login", "verify", "update", "confirm", "password", "bank", "secure")

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
@st.cache_resource
def get_pipeline():
    if not MODEL_PATH.exists():
        st.error("Model file not found. Train it first:\n\n`python train_ui_model.py`")
        st.stop()
    return load_pipeline(str(MODEL_PATH))

def find_keywords(text: str):
    t = text.lower()
    return sorted({kw for kw in SUSPICIOUS_KWS if kw in t})

def read_eml_bytes(data: bytes) -> str:
    """Parse minimal subject + body from .eml bytes; fallback to utf-8 text."""
    from email import policy
    from email.parser import BytesParser
    try:
        msg = BytesParser(policy=policy.default).parsebytes(data)
        parts = []
        subj = msg.get("subject", "")
        if subj:
            parts.append(str(subj))
        body = ""
        if msg.is_multipart():
            plains, htmls = [], []
            for part in msg.walk():
                ct = part.get_content_type()
                if ct == "text/plain":
                    try: plains.append(part.get_content())
                    except Exception: pass
                elif ct == "text/html":
                    try: htmls.append(part.get_content())
                    except Exception: pass
            body = "\n".join(plains) if plains else "\n".join(htmls)
        else:
            ct = msg.get_content_type()
            if ct == "text/plain":
                body = msg.get_content()
            elif ct == "text/html":
                body = msg.get_content()
        text = (("\n".join(parts + [body])) if body else "\n".join(parts)).strip()
        return text
    except Exception:
        # not a real EML, try decode to text
        try:
            return data.decode("utf-8", errors="ignore")
        except Exception:
            return ""

pipe = get_pipeline()

# -----------------------------------------------------------------------------
# Sidebar — model/data info
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Model & Data")
    st.write("Model file:", f"`{MODEL_PATH.name}`")
    if DATA_PATH.exists():
        try:
            dhead = pd.read_csv(DATA_PATH, nrows=500)
            st.write("Dataset sample rows:", len(dhead))
            if "label" in dhead.columns:
                c1 = int(dhead["label"].sum())
                st.write("Class balance — phish(1):", c1, " | ham(0):", len(dhead) - c1)
        except Exception:
            st.write("Dataset info unavailable.")
    st.markdown("---")
    if REPORTS_DIR.joinpath("results.csv").exists():
        st.markdown("**Latest results**")
        try:
            res = pd.read_csv(REPORTS_DIR / "results.csv")
            st.dataframe(res, use_container_width=True, height=180)
        except Exception:
            st.write("Could not read results.csv.")
    if REPORTS_DIR.joinpath("roc_curves.png").exists():
        st.image(str(REPORTS_DIR / "roc_curves.png"), caption="ROC curves", use_column_width=True)

# -----------------------------------------------------------------------------
# Main — single email classification
# -----------------------------------------------------------------------------
st.subheader("Single email classification")
c1, c2 = st.columns([2, 1])

with c1:
    email_text = st.text_area("Email text (subject + body)", height=260, placeholder="Paste email content here…")
    threshold = st.slider("Decision threshold", 0.10, 0.90, 0.50, 0.01,
                          help="Lower to catch more phishing (higher recall). Raise to reduce false alarms.")

    if st.button("Predict", type="primary"):
        if not email_text.strip():
            st.warning("Paste some email text first.")
        else:
            prob = pipe.predict_proba_one(email_text)
            pred = int(prob >= threshold)

            st.metric("Phishing probability", f"{prob:.3f}")
            (st.error if pred == 1 else st.success)("Label: Phishing (1)" if pred == 1 else "Label: Ham / Legit (0)")

            st.markdown("**Extracted URLs**")
            urls = extract_urls(email_text)
            if urls:
                for u in urls: st.write("•", u)
            else:
                st.write("— none —")

            st.markdown("**Suspicious keywords**")
            hits = find_keywords(email_text)
            st.write(", ".join(hits) if hits else "— none —")

with c2:
    st.markdown("**Explainability**")
    try:
        # Top n-grams from Logistic Regression coefficients
        vec = pipe.vec; clf = pipe.clf
        if hasattr(clf, "coef_"):
            coef = clf.coef_[0]; vocab = vec.get_feature_names_out()
            top_pos = np.argsort(coef)[-8:][::-1]
            top_neg = np.argsort(coef)[:8]
            st.caption("Top positive n-grams (→ phishing)")
            st.table(pd.DataFrame({"ngram": vocab[top_pos], "coef": coef[top_pos]}))
            st.caption("Top negative n-grams (→ ham)")
            st.table(pd.DataFrame({"ngram": vocab[top_neg], "coef": coef[top_neg]}))
    except Exception as e:
        st.write("Explainability unavailable:", e)

    # Show URL feature vector values
    st.caption("URL feature values")
    try:
        ufeats = pipe.url.transform([email_text])[0] if email_text.strip() else {}
        if ufeats:
            st.table(pd.DataFrame([ufeats]))
        else:
            st.write("— enter text to compute —")
    except Exception:
        st.write("— not available —")

# -----------------------------------------------------------------------------
# Batch / .eml upload
# -----------------------------------------------------------------------------
st.subheader("Batch classify / .eml upload")
files = st.file_uploader("Upload .eml files or a CSV with a 'body' column", accept_multiple_files=True)
if files:
    rows = []
    for f in files:
        name = f.name
        data = f.read()
        if name.lower().endswith(".eml"):
            txt = read_eml_bytes(data)
        elif name.lower().endswith(".csv"):
            try:
                df_in = pd.read_csv(io.StringIO(data.decode("utf-8", errors="ignore")))
                for _, r in df_in.iterrows():
                    body = str(r.get("body", "")).strip()
                    if body:
                        p = pipe.predict_proba_one(body)
                        rows.append({"source": name, "prob": p, "pred": int(p >= 0.5), "body": body[:120] + ("…" if len(body) > 120 else "")})
                continue
            except Exception:
                txt = data.decode("utf-8", errors="ignore")
        else:
            txt = data.decode("utf-8", errors="ignore")
        p = pipe.predict_proba_one(txt) if txt else 0.0
        rows.append({"source": name, "prob": p, "pred": int(p >= 0.5), "body": (txt[:120] + ("…" if len(txt) > 120 else ""))})
    out = pd.DataFrame(rows)
    st.dataframe(out, use_container_width=True)
    st.download_button("Download results CSV", out.to_csv(index=False).encode("utf-8"),
                       "predictions.csv", "text/csv")

