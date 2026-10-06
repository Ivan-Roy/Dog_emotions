/*
 * app.js — wire the mic to the model.
 * Listens continuously, detects a bark by a volume spike, extracts features
 * (features.js), runs the model (model.js), and shows the translation.
 */
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const orb = $("orb"), hint = $("hint"), glyph = $("glyph");
  const resultEl = $("result"), empty = $("empty");
  const sens = $("sens"), toastEl = $("toast");

  let model = null;
  let audioCtx = null, source = null, processor = null, stream = null;
  let listening = false;

  // bark-detection state
  let recording = false, captured = [], silenceBlocks = 0, preRoll = [];
  const STOP_RATIO = 0.4;        // stop threshold = start * this
  const HANGOVER_BLOCKS = 6;     // ~blocks of quiet before a bark is "done"
  const PREROLL_BLOCKS = 3;      // keep a little audio before onset
  const MIN_DUR = 0.12, MAX_DUR = 2.0; // seconds

  function toast(msg, ms = 2600) {
    toastEl.textContent = msg; toastEl.classList.add("show");
    clearTimeout(toast._t); toast._t = setTimeout(() => toastEl.classList.remove("show"), ms);
  }

  // ---- load model ----------------------------------------------------------
  BarkModel.load("model.json")
    .then((m) => {
      model = m;
      if (m.featureNames.length !== 40) {
        toast("Model loaded, but feature count looks off (" + m.featureNames.length + ").");
      }
    })
    .catch(() => {
      toast("Couldn't load model.json. Make sure it sits next to this page.");
      empty.textContent = "No model found. Train one, export it, and put model.json here.";
    });

  // ---- mic / listening -----------------------------------------------------
  async function startListening() {
    if (!model) { toast("Model isn't ready yet."); return; }
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (e) {
      toast("I can't hear anything — allow mic access in your browser.");
      return;
    }
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    if (audioCtx.state === "suspended") await audioCtx.resume();
    source = audioCtx.createMediaStreamSource(stream);

    // ScriptProcessor is deprecated but has the broadest mobile support.
    processor = audioCtx.createScriptProcessor(2048, 1, 1);
    processor.onaudioprocess = (e) => onBlock(e.inputBuffer.getChannelData(0));
    source.connect(processor);
    processor.connect(audioCtx.destination); // required on some browsers to run

    listening = true;
    document.body.classList.add("listening");
    orb.setAttribute("aria-label", "Stop listening");
    glyph.textContent = "👂"; hint.textContent = "Listening…";
    empty.textContent = "Listening. Each bark gets a reading.";
  }

  function stopListening() {
    listening = false;
    document.body.classList.remove("listening");
    orb.setAttribute("aria-label", "Start listening for barks");
    glyph.textContent = "🐾"; hint.textContent = "Tap to listen";
    orb.style.setProperty("--level", 0);
    if (processor) processor.disconnect();
    if (source) source.disconnect();
    if (stream) stream.getTracks().forEach((t) => t.stop());
    if (audioCtx) audioCtx.close();
    processor = source = stream = audioCtx = null;
    recording = false; captured = []; preRoll = [];
  }

  function onBlock(block) {
    // live level -> orb
    let sum = 0;
    for (let i = 0; i < block.length; i++) sum += block[i] * block[i];
    const rms = Math.sqrt(sum / block.length);
    const start = parseFloat(sens.value);
    orb.parentElement.style.setProperty("--level", Math.min(1, rms / (start * 2)));

    const copy = Float32Array.from(block);

    if (!recording) {
      preRoll.push(copy);
      if (preRoll.length > PREROLL_BLOCKS) preRoll.shift();
      if (rms > start) {
        recording = true;
        captured = preRoll.slice();   // include pre-roll so we don't clip onset
        preRoll = [];
        silenceBlocks = 0;
        orb.parentElement.style.setProperty("--level", 1);
      }
    } else {
      captured.push(copy);
      if (rms < start * STOP_RATIO) {
        silenceBlocks++;
        if (silenceBlocks >= HANGOVER_BLOCKS) finishBark();
      } else {
        silenceBlocks = 0;
      }
      // safety cap
      const dur = captured.length * block.length / audioCtx.sampleRate;
      if (dur > MAX_DUR) finishBark();
    }
  }

  function finishBark() {
    const blocks = captured;
    recording = false; captured = []; silenceBlocks = 0;
    if (!blocks.length) return;

    const total = blocks.reduce((n, b) => n + b.length, 0);
    const dur = total / audioCtx.sampleRate;
    if (dur < MIN_DUR) return; // too short — probably a click, not a bark

    const samples = new Float32Array(total);
    let off = 0;
    for (const b of blocks) { samples.set(b, off); off += b.length; }

    try {
      const { vector } = BarkFeatures.extract(samples, audioCtx.sampleRate);
      const pred = model.predict(vector);
      render(BarkModel.translate(pred));
    } catch (err) {
      toast("Couldn't read that one. Try again.");
    }
  }

  // ---- render --------------------------------------------------------------
  function render(t) {
    empty.style.display = "none";
    resultEl.dataset.valence = t.valence;
    $("headline").textContent = t.headline;
    $("quote").textContent = "\u201C" + t.line + "\u201D";
    $("arousal").textContent = t.arousal;
    $("valence").textContent = t.valence;
    $("hedge").textContent = t.hedge + " · " + Math.round(t.confidence * 100) + "%";
    resultEl.classList.remove("show");
    void resultEl.offsetWidth; // restart the reveal animation
    resultEl.classList.add("show");
  }

  orb.addEventListener("click", () => (listening ? stopListening() : startListening()));
})();
