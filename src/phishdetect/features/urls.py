from __future__ import annotations
import re
from dataclasses import dataclass
import numpy as np

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_SUSPICIOUS_KWS = ("login", "verify", "update", "confirm", "password", "bank", "secure")

def extract_urls(text: str) -> list[str]:
    if not text:
        return []
    return [m.group(0) for m in _URL_RE.finditer(text)]

@dataclass
class URLFeaturizer:
    """Very small numeric feature vector from URLs in text."""
    def transform(self, texts: list[str]) -> np.ndarray:
        feats = []
        for t in texts:
            urls = extract_urls(t or "")
            count = len(urls)
            lengths = [len(u) for u in urls]
            mean_len = float(np.mean(lengths)) if lengths else 0.0
            suspicious_hits = sum(
                any(kw in u.lower() for kw in _SUSPICIOUS_KWS) for u in urls
            )
            dots = sum(u.count(".") for u in urls)
            feats.append([count, mean_len, suspicious_hits, dots])
        return np.asarray(feats, dtype=float)

    @property
    def feature_names(self) -> list[str]:
        return ["url_count", "url_mean_len", "url_suspicious_kw", "url_dot_total"]

