/*
 * model.js — run the exported RandomForest in the browser, then apply the same
 * playful translation layer as barktrans/predict.py.
 *
 * BarkModel.load(url) -> model
 * model.predict(featureVector) -> { arousal:{label,confidence}, valence:{...} }
 * BarkModel.translate(prediction) -> { headline, line, hedge, confidence, ... }
 */
(function (root) {
  "use strict";

  function walkTree(tree, x) {
    let node = 0;
    while (tree.children_left[node] !== -1) {
      node = x[tree.feature[node]] <= tree.threshold[node]
        ? tree.children_left[node]
        : tree.children_right[node];
    }
    return tree.value[node]; // class-probability array at the leaf
  }

  function predictLabel(labelModel, x) {
    const nClasses = labelModel.classes.length;
    const agg = new Array(nClasses).fill(0);
    for (const tree of labelModel.trees) {
      const proba = walkTree(tree, x);
      for (let c = 0; c < nClasses; c++) agg[c] += proba[c];
    }
    let best = 0;
    for (let c = 1; c < nClasses; c++) if (agg[c] > agg[best]) best = c;
    return {
      label: labelModel.classes[best],
      confidence: agg[best] / labelModel.trees.length,
    };
  }

  async function load(url) {
    const res = await fetch(url);
    const json = await res.json();
    return makeModel(json);
  }

  function makeModel(json) {
    return {
      json,
      params: json.params,
      labelCols: json.label_cols,
      featureNames: json.feature_names,
      predict(x) {
        const out = {};
        for (const col of json.label_cols) out[col] = predictLabel(json.labels[col], x);
        return out;
      },
    };
  }

  // ---- translation layer (mirrors predict.py) ------------------------------
  const TRANSLATIONS = {
    "high|positive":   ["\uD83C\uDF89 EXCITED", "This is the BEST thing that has ever happened!!"],
    "high|neutral":    ["\uD83D\uDEAA ALERT",   "Something's happening and you need to know RIGHT NOW."],
    "high|negative":   ["\u26A0\uFE0F ALARM",   "I don't like this. Deal with it. Deal with it now."],
    "medium|positive": ["\uD83D\uDE0A HAPPY",   "Hey. Hey. I like you. Pay attention to me."],
    "medium|neutral":  ["\uD83D\uDC40 CURIOUS", "Noted. Filing this under Things To Monitor."],
    "medium|negative": ["\uD83D\uDE1F UNEASY",  "I'm not sure about this and I'd like it to stop."],
    "low|positive":    ["\uD83D\uDE0C CONTENT", "All is well. Carry on. Possibly a treat though?"],
    "low|neutral":     ["\uD83D\uDC3E MEH",      "Just letting you know I exist."],
    "low|negative":    ["\uD83D\uDE14 DOWN",     "I'd rather you didn't leave."],
  };

  function bucket(label, kind) {
    const s = String(label).toLowerCase();
    if (kind === "arousal") {
      if (["high", "3", "active", "aroused"].some((k) => s.includes(k))) return "high";
      if (["low", "1", "calm", "relax"].some((k) => s.includes(k))) return "low";
      return "medium";
    }
    if (["pos", "happy", "3"].some((k) => s.includes(k))) return "positive";
    if (["neg", "sad", "angry", "1"].some((k) => s.includes(k))) return "negative";
    return "neutral";
  }

  function translate(prediction) {
    const arousal = bucket(prediction.arousal ? prediction.arousal.label : "medium", "arousal");
    const valence = bucket(prediction.valence ? prediction.valence.label : "neutral", "valence");
    const [headline, line] = TRANSLATIONS[`${arousal}|${valence}`] || ["\uD83D\uDC15 BARK", "A classic."];
    const confs = Object.values(prediction).map((v) => v.confidence);
    const avg = confs.length ? confs.reduce((a, b) => a + b, 0) / confs.length : 0;
    return {
      headline, line, arousal, valence, confidence: avg,
      hedge: avg > 0.6 ? "pretty sure" : avg > 0.4 ? "best guess" : "wild guess",
      raw: prediction,
    };
  }

  root.BarkModel = { load, makeModel, translate };
  if (typeof module !== "undefined" && module.exports) module.exports = root.BarkModel;
})(typeof window !== "undefined" ? window : globalThis);
