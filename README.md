# barktrans 🐕

A bark "translator" you can open on your phone. It listens to a dog bark,
classifies its **emotional tone**, and gives a playful interpretation on top.

The model trains in Python; the phone app is pure static HTML/JS, so it hosts
free on GitHub Pages with no server.

## The honest part (read this first)

There is no scientific dictionary mapping a bark to an English sentence. What
barks *do* have is measurable emotional content, described along two axes:

- **Arousal** — how activated (low / medium / high)
- **Valence** — emotional tone (negative / neutral / positive)

This project predicts arousal + valence, then maps that to a fun "translation."
The emotion reading is the real part; the phrases are entertainment. It's a mood
ring, not a decoder ring. Published studies land around 40–55% accuracy on this
kind of task — well above chance, far from perfect. Demand / "wants something"
barks are the hardest to pin down acoustically.

## How it fits together

```
 Python (your machine)                 Browser (your phone)
 ─────────────────────                 ────────────────────
 features.py  ─ wav → 40 features      features.js ─ same 40 features
 train.py     ─ dataset → model        model.js    ─ runs the model
 export_model ─ model → model.json ──► app.js      ─ mic → bark → translation
```

`features.js` is a line-for-line port of `features.py`, verified to match to
floating-point precision. If you ever change `features.py`, re-check parity.

## 1. Set up and train (Python)

```bash
pip install -r requirements.txt
```

Get an emotion-labeled dataset — easiest is **Barkopedia Dog Emotion
Classification** on Hugging Face (arousal + valence labels). You need a folder
of `.wav` clips and a CSV with a filename column and label column(s).

```bash
python -m barktrans.train \
    --audio-dir data/husky_train \
    --labels data/husky_train_labels.csv \
    --audio-col audio_id \
    --label-cols arousal,valence \
    --out models/bark_model.joblib
```

It prints honest cross-validated accuracy per label.

## 2. Export for the web

```bash
python -m barktrans.export_model \
    --model models/bark_model.joblib \
    --sample data/husky_train/any_clip.wav \
    --out-dir docs
```

This writes `docs/model.json` (the model) and `docs/reference.json` (a self-test
vector). The repo already ships with a demo model so the app runs before you
train your own — just overwrite it with this command.

## 3. Try it locally

```bash
python -m http.server --directory docs 8000
```

Open http://localhost:8000 — mic access works on `localhost` without HTTPS. Tap
the orb and bark at it (or play a bark clip).

## 4. Host on GitHub Pages (for your phone)

```bash
git add -A && git commit -m "barktrans"
git remote add origin https://github.com/YOU/barktrans.git
git push -u origin main
```

Then on GitHub: **Settings → Pages → Build and deployment → Source: Deploy from
a branch → Branch: `main`, Folder: `/docs` → Save.**

After a minute your app is live at `https://YOU.github.io/barktrans/`. Open that
on your phone (it's HTTPS, so the mic works), allow mic access, and optionally
"Add to Home Screen" for an app-like icon.

## Project layout

```
barktrans/          Python: the model pipeline
  features.py       wav → 40-feature vector
  train.py          dataset → trained model (+ accuracy)
  predict.py        model + wav → prediction + translation (CLI)
  export_model.py   model → docs/model.json for the browser
docs/               the static phone app (GitHub Pages serves this)
  index.html  app.js  features.js  model.js  model.json  reference.json
requirements.txt
```

## Tuning on your phone

If barks are missed or random noise triggers readings, use the **Sensitivity**
slider that appears while listening. The orb reacts to live mic level so you can
see what counts as "loud enough."
