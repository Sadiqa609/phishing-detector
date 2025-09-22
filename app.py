from __future__ import annotations
from pathlib import Path
import io
import numpy as np
import pandas as pd
import streamlit as st

from phishdetect.predictor import load_pipeline
from phishdetect.features.urls import extract_urls

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
        txt = (("\n".join(parts + [body])) if body else "\n".join(parts)).strip()
        return txt
    except Exception:
        try:
            return data.decode("utf-8", errors="ignore")
        except Exception:
            return ""

pipe = get_pipeline()

# -----------------------------------------------------------------------------
# Sidebar — model/data info and saved results
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Model & Data")
    st.write("Model file:", f"`{MODEL_PATH.name}`")
    if DATA_PATH.exists():
        try:
            dhead = pd.read_csv(DATA_PATH, nrows=500)
            st.write("Sample rows:", len(dhead))
            if "label" in dhead.columns:
                c1 = int(dhead["label"].sum())
                st.write("Class balance — phish(1):", c1, " | ham(0):", len(dhead) - c1)
        except Exception:
            st.write("Dataset info unavailable.")
    st.markdown("---")
    if (REPORTS_DIR / "results.csv").exists():
        st.markdown("**Latest results (test)**")
        try:
            res = pd.read_csv(REPORTS_DIR / "results.csv")
            st.dataframe(res, use_container_width=True, height=180)
        except Exception:
            st.write("Could not read results.csv.")
    if (REPORTS_DIR / "roc_curves.png").exists():
        st.image(str(REPORTS_DIR / "roc_curves.png"), caption="ROC curves", use_column_width=True)

# -----------------------------------------------------------------------------
# Main — single email classification
# -----------------------------------------------------------------------------
st.subheader("Single email classification")
c1, c2 = st.columns([2, 1])

with c1:
    email_text = st.text_area("Email text (subject + body)", height=260, placeholder="Paste email content here…")
    threshold = st.slider(
        "Decision threshold", 0.10, 0.90, 0.50, 0.01,
        help="Lower to catch more phishing (higher recall). Raise to reduce false alarms."
    )

    if st.button("Predict", type="primary"):
        if not email_text.strip():
            st.warning("Paste some email text first.")
        else:
            prob = pipe.predict_proba_one(email_text)
            label = "Phishing" if prob >= threshold else "Ham"

            st.metric("Phishing probability", f"{prob:.3f}")
            (st.error if label == "Phishing" else st.success)(f"Label: {label}")

            st.markdown("**Extracted URLs**")
            urls = extract_urls(email_text)
            if urls:
                for u in urls: st.write("•", u)
            else:
                st.write("— none —")

            st.markdown("**Suspicious keywords**")
            hits = find_keywords(email_text)
            st.write(", ".join(hits) if hits else "— none —")

            # URL feature values (explainability)
            st.markdown("**URL feature values**")
            try:
                ufeats = pipe.url.transform([email_text])[0]
                st.table(pd.DataFrame([ufeats]))
            except Exception:
                st.write("— not available —")

with c2:
    st.markdown("**Explainability**")
    try:
        # Top n-grams from Logistic Regression coefficients
        vec = pipe.vec
        clf = pipe.clf
        if hasattr(clf, "coef_"):
            coef = clf.coef_[0]
            vocab = vec.get_feature_names_out()
            top_pos = np.argsort(coef)[-8:][::-1]   # strongest phishing indicators
            top_neg = np.argsort(coef)[:8]          # strongest ham indicators
            st.caption("Top phishing n-grams")
            st.table(pd.DataFrame({"ngram": vocab[top_pos], "coef": coef[top_pos]}))
            st.caption("Top ham n-grams")
            st.table(pd.DataFrame({"ngram": vocab[top_neg], "coef": coef[top_neg]}))
    except Exception as e:
        st.write("Explainability unavailable:", e)

    st.caption("URL feature values")
    if 'email_text' in locals() and email_text.strip():
        try:
            ufeats = pipe.url.transform([email_text])[0]
            st.table(pd.DataFrame([ufeats]))
        except Exception:
            st.write("— not available —")
    else:
        st.write("— enter text to compute —")

# -----------------------------------------------------------------------------
# Batch / .eml upload  (upload message + Predict button + badges + CSV download)
# -----------------------------------------------------------------------------
st.subheader("Batch classify / .eml upload")

# 1) User selects files
files = st.file_uploader(
    "Upload .eml files or a CSV with a 'body' column",
    accept_multiple_files=True,
    key="batch_files"
)

# 2) Immediate feedback after upload
status = st.empty()
if files and len(files) > 0:
    status.success(f"✅ {len(files)} file(s) uploaded. Click **Predict** to analyze.")
else:
    status.info("Upload one or more .eml files, or a CSV with a 'body' column, then click Predict.")

# 3) Only process when user clicks Predict
if st.button("Predict", type="primary", help="Run prediction on uploaded files"):
    if not files or len(files) == 0:
        st.warning("Please upload at least one file first.")
    else:
        rows = []
        for f in files:
            name = f.name
            data = f.read()

            # .eml branch
            if name.lower().endswith(".eml"):
                txt = read_eml_bytes(data)

            # CSV branch (expects 'body' column)
            elif name.lower().endswith(".csv"):
                try:
                    df_in = pd.read_csv(io.StringIO(data.decode("utf-8", errors="ignore")))
                    for _, r in df_in.iterrows():
                        body = str(r.get("body", "")).strip()
                        if not body:
                            continue
                        p = pipe.predict_proba_one(body)
                        plain_label = "Phishing" if p >= 0.5 else "Ham"
                        rows.append({
                            "label": plain_label,  # plain text for CSV download
                            "label_html": (
                                f"<span style='color:white; background-color:#d9534f; padding:2px 6px; border-radius:4px;'>Phishing</span>"
                                if plain_label == "Phishing"
                                else f"<span style='color:white; background-color:#5cb85c; padding:2px 6px; border-radius:4px;'>Ham</span>"
                            ),
                            "prob": p,
                            "source": name,
                            "body": body[:120] + ("…" if len(body) > 120 else "")
                        })
                    # move on to next uploaded file
                    continue
                except Exception:
                    # if parsing CSV fails, fall back to plain text
                    txt = data.decode("utf-8", errors="ignore")

            # plain text fallback for anything else
            else:
                txt = data.decode("utf-8", errors="ignore")

            # Single-text path (eml/plain)
            p = pipe.predict_proba_one(txt) if txt else 0.0
            plain_label = "Phishing" if p >= 0.5 else "Ham"
            rows.append({
                "label": plain_label,
                "label_html": (
                    f"<span style='color:white; background-color:#d9534f; padding:2px 6px; border-radius:4px;'>Phishing</span>"
                    if plain_label == "Phishing"
                    else f"<span style='color:white; background-color:#5cb85c; padding:2px 6px; border-radius:4px;'>Ham</span>"
                ),
                "prob": p,
                "source": name,
                "body": txt[:120] + ("…" if len(txt) > 120 else "")
            })

        # Assemble results table
        out = pd.DataFrame(rows)

        # Order columns for display & CSV
        csv_out = out[["label", "prob", "source", "body"]]
        display_out = out[["label_html", "prob", "source", "body"]].rename(columns={"label_html": "label"})

        st.success(f"✅ Predictions complete for {len(out)} item(s).")

        # Use HTML to render badges professionally
        st.write(display_out.to_html(index=False, escape=False), unsafe_allow_html=True)

        st.download_button(
            "Download results CSV",
            csv_out.to_csv(index=False).encode("utf-8"),
            "predictions.csv",
            "text/csv"
        )

