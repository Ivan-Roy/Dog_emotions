"""
features.py — turn a bark .wav into a fixed-length numeric feature vector.

Self-contained: depends only on numpy + scipy, so it runs anywhere without
librosa. If you later `pip install librosa`, you can swap in its richer
loaders/MFCCs, but these features are plenty for a weekend bark classifier.

The public entry point is `extract_features(path)` -> (vector, names).
Every clip, whatever its length, maps to the SAME vector, so the model
always sees a consistent shape.
"""

from __future__ import annotations

import numpy as np
from scipy.io import wavfile

# ---- config -----------------------------------------------------------------
TARGET_SR = 22050          # resample everything to this, so features compare fairly
N_MFCC = 13                # number of MFCC coefficients
N_MELS = 40                # mel filterbank size
FRAME_LEN = 2048           # FFT window (samples)
HOP_LEN = 512              # step between frames (samples)
FMIN = 50.0                # lowest mel band (Hz) — dogs don't bark below this
FMAX = TARGET_SR / 2       # Nyquist


# ---- loading ----------------------------------------------------------------
def load_wav(path: str, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """Load a wav as mono float32 in [-1, 1], resampled to target_sr."""
    sr, data = wavfile.read(path)

    # scipy returns ints for PCM wavs — normalize by dtype range to float [-1, 1]
    if np.issubdtype(data.dtype, np.integer):
        max_val = np.iinfo(data.dtype).max
        data = data.astype(np.float32) / max_val
    else:
        data = data.astype(np.float32)

    # mix down to mono
    if data.ndim > 1:
        data = data.mean(axis=1)

    # resample via linear interpolation if needed.
    # (Linear, not FFT-based, so the JavaScript port can match it exactly.)
    if sr != target_sr and len(data) > 1:
        n_target = int(round(len(data) * target_sr / sr))
        if n_target > 0:
            old_idx = np.arange(len(data))
            new_idx = np.linspace(0, len(data) - 1, n_target)
            data = np.interp(new_idx, old_idx, data).astype(np.float32)
        sr = target_sr

    return data, sr


# ---- framing ----------------------------------------------------------------
def _frame(signal: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    """Slice signal into overlapping frames -> shape (n_frames, frame_len)."""
    if len(signal) < frame_len:
        signal = np.pad(signal, (0, frame_len - len(signal)))
    n_frames = 1 + (len(signal) - frame_len) // hop
    idx = np.arange(frame_len)[None, :] + hop * np.arange(n_frames)[:, None]
    return signal[idx]


# ---- mel filterbank + MFCC --------------------------------------------------
def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """Build a (n_mels, n_fft//2+1) triangular mel filterbank."""
    mel_pts = np.linspace(_hz_to_mel(np.array([fmin]))[0],
                          _hz_to_mel(np.array([fmax]))[0], n_mels + 2)
    hz_pts = _mel_to_hz(mel_pts)
    bins = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        if center == left:
            center = left + 1
        if right == center:
            right = center + 1
        for k in range(left, center):
            if center != left:
                fb[m - 1, k] = (k - left) / (center - left)
        for k in range(center, right):
            if right != center:
                fb[m - 1, k] = (right - k) / (right - center)
    return fb


def _dct(x: np.ndarray, n_out: int) -> np.ndarray:
    """Type-II DCT of each row of x, keep first n_out coefficients."""
    n = x.shape[1]
    k = np.arange(n_out)[:, None]
    i = np.arange(n)[None, :]
    basis = np.cos(np.pi / n * (i + 0.5) * k)  # (n_out, n)
    return x @ basis.T


def _mfcc(frames: np.ndarray, sr: int) -> np.ndarray:
    """frames (n_frames, frame_len) -> MFCCs (n_frames, N_MFCC)."""
    windowed = frames * np.hanning(frames.shape[1])
    spectrum = np.abs(np.fft.rfft(windowed, axis=1)) ** 2  # power spectrum
    fb = _mel_filterbank(sr, FRAME_LEN, N_MELS, FMIN, FMAX)
    mel_energy = spectrum @ fb.T                            # (n_frames, n_mels)
    log_mel = np.log(mel_energy + 1e-10)
    return _dct(log_mel, N_MFCC)


# ---- scalar spectral features ----------------------------------------------
def _spectral_features(frames: np.ndarray, sr: int) -> dict[str, np.ndarray]:
    windowed = frames * np.hanning(frames.shape[1])
    mag = np.abs(np.fft.rfft(windowed, axis=1))
    freqs = np.fft.rfftfreq(FRAME_LEN, 1.0 / sr)
    mag_sum = mag.sum(axis=1) + 1e-10

    centroid = (mag * freqs).sum(axis=1) / mag_sum
    bandwidth = np.sqrt((((freqs - centroid[:, None]) ** 2) * mag).sum(axis=1) / mag_sum)

    # rolloff: freq below which 85% of energy sits
    cumulative = np.cumsum(mag, axis=1)
    threshold = 0.85 * cumulative[:, -1:]
    rolloff_idx = (cumulative >= threshold).argmax(axis=1)
    rolloff = freqs[rolloff_idx]

    # flatness: geometric mean / arithmetic mean of the spectrum (tonal vs noisy)
    geo = np.exp(np.mean(np.log(mag + 1e-10), axis=1))
    flatness = geo / (mag.mean(axis=1) + 1e-10)

    zcr = np.mean(np.abs(np.diff(np.sign(frames), axis=1)) > 0, axis=1)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))

    return {
        "centroid": centroid,
        "bandwidth": bandwidth,
        "rolloff": rolloff,
        "flatness": flatness,
        "zcr": zcr,
        "rms": rms,
    }


def _f0_autocorr(signal: np.ndarray, sr: int) -> float:
    """Rough fundamental-frequency estimate via autocorrelation (whole clip)."""
    if len(signal) < FRAME_LEN:
        return 0.0
    sig = signal - signal.mean()
    corr = np.correlate(sig, sig, mode="full")[len(sig) - 1:]
    min_lag = int(sr / 1000)   # cap pitch at 1000 Hz
    max_lag = int(sr / 50)     # floor pitch at 50 Hz
    if max_lag >= len(corr):
        return 0.0
    segment = corr[min_lag:max_lag]
    if len(segment) == 0 or segment.max() <= 0:
        return 0.0
    peak = segment.argmax() + min_lag
    return float(sr / peak) if peak > 0 else 0.0


# ---- public API -------------------------------------------------------------
def extract_features(path: str) -> tuple[np.ndarray, list[str]]:
    """Return (feature_vector, feature_names) for one wav file."""
    signal, sr = load_wav(path)
    if len(signal) == 0:
        raise ValueError(f"Empty audio: {path}")

    frames = _frame(signal, FRAME_LEN, HOP_LEN)

    mfccs = _mfcc(frames, sr)                 # (n_frames, N_MFCC)
    spec = _spectral_features(frames, sr)     # dict of (n_frames,)

    values: list[float] = []
    names: list[str] = []

    # MFCCs: mean + std across time
    for i in range(N_MFCC):
        values += [float(mfccs[:, i].mean()), float(mfccs[:, i].std())]
        names += [f"mfcc{i}_mean", f"mfcc{i}_std"]

    # spectral scalars: mean + std across time
    for key, arr in spec.items():
        values += [float(arr.mean()), float(arr.std())]
        names += [f"{key}_mean", f"{key}_std"]

    # clip-level features
    values += [float(len(signal) / sr), _f0_autocorr(signal, sr)]
    names += ["duration_s", "f0_hz"]

    return np.array(values, dtype=np.float32), names


if __name__ == "__main__":
    import sys
    vec, names = extract_features(sys.argv[1])
    for n, v in zip(names, vec):
        print(f"{n:16s} {v: .4f}")
    print(f"\n{len(vec)} features")
