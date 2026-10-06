/*
 * features.js — browser port of barktrans/features.py.
 * MUST produce the same 40 features, in the same order, as the Python code,
 * or the model's predictions are meaningless. Verified against reference.json.
 *
 * Entry point: BarkFeatures.extract(samples, sampleRate) -> {vector, names}
 *   samples: Float32Array in [-1, 1], any sample rate (gets resampled internally)
 */
(function (root) {
  "use strict";

  const TARGET_SR = 22050;
  const N_MFCC = 13;
  const N_MELS = 40;
  const FRAME_LEN = 2048;
  const HOP_LEN = 512;
  const FMIN = 50.0;
  const FMAX = TARGET_SR / 2;

  // Python's round() / numpy round half to EVEN, not half up. Match it, or the
  // resampled length (and therefore every feature) drifts by a sample.
  function roundHalfEven(x) {
    const floor = Math.floor(x);
    const diff = x - floor;
    if (Math.abs(diff - 0.5) < 1e-9) return floor % 2 === 0 ? floor : floor + 1;
    return Math.round(x);
  }

  // ---- linear resample (matches numpy np.interp over np.linspace) ----------
  function resample(data, sr, targetSr) {
    if (sr === targetSr || data.length <= 1) return data;
    const nTarget = roundHalfEven((data.length * targetSr) / sr);
    if (nTarget <= 0) return data;
    const out = new Float32Array(nTarget);
    const N = data.length;
    for (let i = 0; i < nTarget; i++) {
      // new_idx = linspace(0, N-1, nTarget)
      const x = nTarget === 1 ? 0 : (i * (N - 1)) / (nTarget - 1);
      const lo = Math.floor(x);
      const hi = Math.min(lo + 1, N - 1);
      const frac = x - lo;
      out[i] = data[lo] * (1 - frac) + data[hi] * frac;
    }
    return out;
  }

  // ---- framing (matches _frame) -------------------------------------------
  function frame(signal, frameLen, hop) {
    let sig = signal;
    if (sig.length < frameLen) {
      const padded = new Float32Array(frameLen);
      padded.set(sig);
      sig = padded;
    }
    const nFrames = 1 + Math.floor((sig.length - frameLen) / hop);
    const frames = [];
    for (let f = 0; f < nFrames; f++) {
      frames.push(sig.subarray(f * hop, f * hop + frameLen));
    }
    return frames;
  }

  // ---- iterative radix-2 FFT (in place, complex) --------------------------
  function fft(re, im) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        [re[i], re[j]] = [re[j], re[i]];
        [im[i], im[j]] = [im[j], im[i]];
      }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = (-2 * Math.PI) / len;
      const wRe = Math.cos(ang), wIm = Math.sin(ang);
      for (let i = 0; i < n; i += len) {
        let curRe = 1, curIm = 0;
        for (let k = 0; k < len / 2; k++) {
          const a = i + k, b = i + k + len / 2;
          const tRe = re[b] * curRe - im[b] * curIm;
          const tIm = re[b] * curIm + im[b] * curRe;
          re[b] = re[a] - tRe; im[b] = im[a] - tIm;
          re[a] += tRe; im[a] += tIm;
          const nextRe = curRe * wRe - curIm * wIm;
          curIm = curRe * wIm + curIm * wRe;
          curRe = nextRe;
        }
      }
    }
  }

  // magnitude + power spectra (first n/2+1 bins), with Hann window
  function spectra(frameArr) {
    const n = frameArr.length;
    const re = new Float64Array(n), im = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const w = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1)); // np.hanning
      re[i] = frameArr[i] * w;
    }
    fft(re, im);
    const half = n / 2 + 1;
    const mag = new Float64Array(half), pow = new Float64Array(half);
    for (let i = 0; i < half; i++) {
      mag[i] = Math.hypot(re[i], im[i]);
      pow[i] = mag[i] * mag[i];
    }
    return { mag, pow };
  }

  // ---- mel filterbank (matches _mel_filterbank) ---------------------------
  const hzToMel = (hz) => 2595.0 * Math.log10(1.0 + hz / 700.0);
  const melToHz = (mel) => 700.0 * (Math.pow(10.0, mel / 2595.0) - 1.0);

  function melFilterbank(sr, nFft, nMels, fmin, fmax) {
    const mLow = hzToMel(fmin), mHigh = hzToMel(fmax);
    const melPts = [];
    for (let i = 0; i < nMels + 2; i++) {
      melPts.push(mLow + ((mHigh - mLow) * i) / (nMels + 1));
    }
    const half = nFft / 2 + 1;
    const bins = melPts.map((m) => {
      let b = Math.floor(((nFft + 1) * melToHz(m)) / sr);
      return Math.max(0, Math.min(b, nFft / 2));
    });
    const fb = [];
    for (let m = 1; m <= nMels; m++) {
      const row = new Float64Array(half);
      let left = bins[m - 1], center = bins[m], right = bins[m + 1];
      if (center === left) center = left + 1;
      if (right === center) right = center + 1;
      for (let k = left; k < center; k++) {
        if (center !== left) row[k] = (k - left) / (center - left);
      }
      for (let k = center; k < right; k++) {
        if (right !== center) row[k] = (right - k) / (right - center);
      }
      fb.push(row);
    }
    return fb;
  }

  // ---- DCT-II (matches _dct) ----------------------------------------------
  function dct(row, nOut) {
    const n = row.length;
    const out = new Float64Array(nOut);
    for (let k = 0; k < nOut; k++) {
      let s = 0;
      for (let i = 0; i < n; i++) {
        s += row[i] * Math.cos((Math.PI / n) * (i + 0.5) * k);
      }
      out[k] = s;
    }
    return out;
  }

  const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
  const std = (a) => {
    const m = mean(a);
    return Math.sqrt(a.reduce((x, y) => x + (y - m) * (y - m), 0) / a.length);
  };

  // ---- f0 via autocorrelation (matches _f0_autocorr) ----------------------
  function f0Autocorr(signal, sr) {
    if (signal.length < FRAME_LEN) return 0.0;
    const m = mean(signal);
    const sig = new Float64Array(signal.length);
    for (let i = 0; i < signal.length; i++) sig[i] = signal[i] - m;
    const minLag = Math.floor(sr / 1000);
    const maxLag = Math.floor(sr / 50);
    if (maxLag >= sig.length) return 0.0;
    let bestVal = -Infinity, bestLag = -1;
    for (let lag = minLag; lag < maxLag; lag++) {
      let s = 0;
      for (let n = 0; n + lag < sig.length; n++) s += sig[n] * sig[n + lag];
      if (s > bestVal) { bestVal = s; bestLag = lag; }
    }
    if (bestVal <= 0 || bestLag <= 0) return 0.0;
    return sr / bestLag;
  }

  // ---- public: extract -----------------------------------------------------
  function extract(samplesIn, sampleRate) {
    const signal = resample(samplesIn, sampleRate, TARGET_SR);
    const sr = TARGET_SR;
    if (signal.length === 0) throw new Error("Empty audio");

    const frames = frame(signal, FRAME_LEN, HOP_LEN);
    const fb = melFilterbank(sr, FRAME_LEN, N_MELS, FMIN, FMAX);
    const freqs = [];
    for (let i = 0; i < FRAME_LEN / 2 + 1; i++) freqs.push((i * sr) / FRAME_LEN);

    // per-frame accumulators
    const mfccCols = Array.from({ length: N_MFCC }, () => []);
    const centroid = [], bandwidth = [], rolloff = [], flatness = [], zcr = [], rms = [];

    for (const fr of frames) {
      const { mag, pow } = spectra(fr);

      // MFCC: mel energy -> log -> DCT
      const logMel = new Float64Array(N_MELS);
      for (let m = 0; m < N_MELS; m++) {
        let e = 0;
        const row = fb[m];
        for (let k = 0; k < pow.length; k++) e += pow[k] * row[k];
        logMel[m] = Math.log(e + 1e-10);
      }
      const mf = dct(logMel, N_MFCC);
      for (let i = 0; i < N_MFCC; i++) mfccCols[i].push(mf[i]);

      // spectral scalars (use magnitude)
      let magSum = 1e-10, cen = 0;
      for (let k = 0; k < mag.length; k++) { magSum += mag[k]; cen += mag[k] * freqs[k]; }
      cen /= magSum;
      centroid.push(cen);
      let bw = 0;
      for (let k = 0; k < mag.length; k++) bw += (freqs[k] - cen) ** 2 * mag[k];
      bandwidth.push(Math.sqrt(bw / magSum));

      let cum = 0, total = 0;
      for (let k = 0; k < mag.length; k++) total += mag[k];
      const thr = 0.85 * total;
      let rollIdx = 0;
      for (let k = 0; k < mag.length; k++) { cum += mag[k]; if (cum >= thr) { rollIdx = k; break; } }
      rolloff.push(freqs[rollIdx]);

      let logSum = 0, arith = 0;
      for (let k = 0; k < mag.length; k++) { logSum += Math.log(mag[k] + 1e-10); arith += mag[k]; }
      const geo = Math.exp(logSum / mag.length);
      flatness.push(geo / (arith / mag.length + 1e-10));

      let zc = 0;
      const sign = (x) => (x > 0 ? 1 : x < 0 ? -1 : 0);
      for (let i = 1; i < fr.length; i++) if (Math.abs(sign(fr[i]) - sign(fr[i - 1])) > 0) zc++;
      zcr.push(zc / (fr.length - 1));

      let sq = 0;
      for (let i = 0; i < fr.length; i++) sq += fr[i] * fr[i];
      rms.push(Math.sqrt(sq / fr.length));
    }

    const values = [], names = [];
    for (let i = 0; i < N_MFCC; i++) {
      values.push(mean(mfccCols[i]), std(mfccCols[i]));
      names.push(`mfcc${i}_mean`, `mfcc${i}_std`);
    }
    const specs = { centroid, bandwidth, rolloff, flatness, zcr, rms };
    for (const key of Object.keys(specs)) {
      values.push(mean(specs[key]), std(specs[key]));
      names.push(`${key}_mean`, `${key}_std`);
    }
    values.push(signal.length / sr, f0Autocorr(signal, sr));
    names.push("duration_s", "f0_hz");

    return { vector: values, names };
  }

  root.BarkFeatures = { extract, TARGET_SR, FRAME_LEN, HOP_LEN };
  if (typeof module !== "undefined" && module.exports) module.exports = root.BarkFeatures;
})(typeof window !== "undefined" ? window : globalThis);
