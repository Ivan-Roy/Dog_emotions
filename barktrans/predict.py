"""
predict.py — load a trained model and interpret a bark.

Returns both the honest prediction (arousal/valence + confidence) and the
playful "translation" layer on top. Reused by the CLI and the web app.
"""

from __future__ import annotations

import argparse

import joblib
import numpy as np

from .features import extract_features


def load_model(path: str) -> dict:
    return joblib.load(path)


def predict_vector(bundle: dict, vec: np.ndarray) -> dict:
    """Predict every trained label for one feature vector."""
    out = {}
    X = vec.reshape(1, -1)
    for col in bundle["label_cols"]:
        clf, le = bundle["models"][col], bundle["encoders"][col]
        proba = clf.predict_proba(X)[0]
        idx = int(proba.argmax())
        out[col] = {
            "label": le.inverse_transform([idx])[0],
            "confidence": float(proba[idx]),
        }
    return out


def predict_file(bundle: dict, path: str) -> dict:
    vec, _ = extract_features(path)
    return predict_vector(bundle, vec)


# ---- the fun layer: map (arousal, valence) -> a playful "translation" --------
# These are intentionally labeled as guesses, not facts. Arousal/valence are
# the honest model outputs; the phrases are the entertainment on top.
TRANSLATIONS = {
    ("high", "positive"): ("🎉 EXCITED", "This is the BEST thing that has ever happened!!"),
    ("high", "neutral"):  ("🚪 ALERT",   "Something's happening and you need to know RIGHT NOW."),
    ("high", "negative"): ("⚠️ ALARM",   "I don't like this. Deal with it. Deal with it now."),
    ("medium", "positive"): ("😊 HAPPY",  "Hey. Hey. I like you. Pay attention to me."),
    ("medium", "neutral"):  ("👀 CURIOUS", "Noted. Filing this under Things To Monitor."),
    ("medium", "negative"): ("😟 UNEASY",  "I'm not sure about this and I'd like it to stop."),
    ("low", "positive"):   ("😌 CONTENT",  "All is well. Carry on. Possibly a treat though?"),
    ("low", "neutral"):    ("🐾 MEH",      "Just letting you know I exist."),
    ("low", "negative"):   ("😔 DOWN",     "I'd rather you didn't leave."),
}


def _bucket(label: str, kind: str) -> str:
    """Normalize whatever the dataset called it into high/med/low or pos/neu/neg."""
    s = str(label).lower()
    if kind == "arousal":
        if any(k in s for k in ("high", "3", "active", "aroused")):
            return "high"
        if any(k in s for k in ("low", "1", "calm", "relax")):
            return "low"
        return "medium"
    # valence
    if any(k in s for k in ("pos", "happy", "3")):
        return "positive"
    if any(k in s for k in ("neg", "sad", "angry", "1")):
        return "negative"
    return "neutral"


def translate(prediction: dict) -> dict:
    """Turn model output into a playful headline + line, with a confidence note."""
    arousal = _bucket(prediction.get("arousal", {}).get("label", "medium"), "arousal")
    valence = _bucket(prediction.get("valence", {}).get("label", "neutral"), "valence")
    headline, line = TRANSLATIONS.get((arousal, valence), ("🐕 BARK", "A classic."))

    confs = [v["confidence"] for v in prediction.values() if "confidence" in v]
    avg_conf = float(np.mean(confs)) if confs else 0.0

    return {
        "headline": headline,
        "line": line,
        "arousal": arousal,
        "valence": valence,
        "confidence": avg_conf,
        "hedge": "pretty sure" if avg_conf > 0.6 else "best guess" if avg_conf > 0.4 else "wild guess",
        "raw": prediction,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Interpret a bark.")
    ap.add_argument("wav")
    ap.add_argument("--model", default="models/bark_model.joblib")
    args = ap.parse_args(argv)

    bundle = load_model(args.model)
    pred = predict_file(bundle, args.wav)
    result = translate(pred)

    print(f"\n{result['headline']}  ({result['hedge']}, "
          f"{result['confidence']:.0%} confident)")
    print(f'  "{result["line"]}"')
    print(f"  arousal={result['arousal']}  valence={result['valence']}")
    print("  raw:", {k: (v["label"], round(v["confidence"], 2))
                     for k, v in pred.items()})


if __name__ == "__main__":
    main()
