"""
Audio content verification utility for testing.

Verifies that separated audio stems actually contain what their labels claim
by correlating against known-good reference separations.
"""

import numpy as np
import librosa
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class StemVerification:
    """Result of verifying a single stem's content."""
    file_path: str
    label: str
    corr_vocal: float
    corr_instrumental: float
    corr_mix: float
    rms: float
    detected_content: str
    label_matches: bool


def load_references(input_dir="tests/inputs", sr=44100):
    """Load known-good reference stems and the original mix.

    Returns (ref_vocal, ref_instrumental, ref_mix, min_len) as mono numpy arrays.
    """
    ref_vocal, _ = librosa.load(
        os.path.join(input_dir, "mardy20s_(Vocals)_mel_band_roformer_karaoke_aufr33_viperx_sdr_10.flac"),
        sr=sr, mono=True,
    )
    ref_inst, _ = librosa.load(
        os.path.join(input_dir, "mardy20s_(Instrumental)_mel_band_roformer_karaoke_aufr33_viperx_sdr_10.flac"),
        sr=sr, mono=True,
    )
    ref_mix, _ = librosa.load(
        os.path.join(input_dir, "mardy20s.flac"),
        sr=sr, mono=True,
    )
    min_len = min(len(ref_vocal), len(ref_inst), len(ref_mix))
    return ref_vocal[:min_len], ref_inst[:min_len], ref_mix[:min_len], min_len


def classify_audio(audio_mono, ref_vocal, ref_instrumental, ref_mix, min_len):
    """Classify audio content by correlation against references.

    Returns (corr_vocal, corr_instrumental, corr_mix, rms, detected_content).
    """
    y = audio_mono[:min_len]
    if len(y) < min_len:
        y = np.pad(y, (0, min_len - len(y)))

    corr_vocal = np.corrcoef(y, ref_vocal)[0, 1]
    corr_inst = np.corrcoef(y, ref_instrumental)[0, 1]
    corr_mix = np.corrcoef(y, ref_mix)[0, 1]
    rms = float(np.sqrt(np.mean(y ** 2)))

    if corr_mix > 0.95:
        detected = "FULL_MIX"
    elif rms < 0.005:
        detected = "SILENT"
    elif corr_vocal > corr_inst and corr_vocal > 0.5:
        detected = "VOCALS"
    elif corr_inst > corr_vocal and corr_inst > 0.5:
        detected = "INSTRUMENTAL"
    else:
        detected = "UNCLEAR"

    return corr_vocal, corr_inst, corr_mix, rms, detected


def verify_stem(file_path, label, ref_vocal, ref_instrumental, ref_mix, min_len, sr=44100):
    """Verify a single stem file's content matches its label.

    Args:
        file_path: Path to the audio file.
        label: The stem label (e.g., "Vocals", "Instrumental").
        ref_vocal, ref_instrumental, ref_mix: Reference arrays from load_references().
        min_len: Minimum length for alignment.
        sr: Sample rate.

    Returns:
        StemVerification dataclass with results.
    """
    y, _ = librosa.load(file_path, sr=sr, mono=True)
    cv, ci, cm, rms, detected = classify_audio(y, ref_vocal, ref_instrumental, ref_mix, min_len)

    # Determine if label matches detected content
    label_lower = label.lower()
    if detected == "VOCALS":
        label_matches = "vocal" in label_lower or "karaoke" not in label_lower and label_lower in ("vocals",)
    elif detected == "INSTRUMENTAL":
        label_matches = label_lower in ("instrumental", "karaoke", "inst", "other", "no_vocals")
    elif detected == "FULL_MIX":
        label_matches = False  # A stem should never be the full mix
    elif detected == "SILENT":
        label_matches = False
    else:
        label_matches = False

    return StemVerification(
        file_path=file_path,
        label=label,
        corr_vocal=cv,
        corr_instrumental=ci,
        corr_mix=cm,
        rms=rms,
        detected_content=detected,
        label_matches=label_matches,
    )


def verify_separation_outputs(output_files, ref_vocal, ref_instrumental, ref_mix, min_len, sr=44100):
    """Verify all output files from a separation.

    Args:
        output_files: List of output file paths (with stem names in parentheses).
        ref_vocal, ref_instrumental, ref_mix: Reference arrays.
        min_len: Minimum length for alignment.
        sr: Sample rate.

    Returns:
        List of StemVerification results.
    """
    import re

    results = []
    for fp in output_files:
        fname = os.path.basename(fp)
        match = re.search(r'_\(([^)]+)\)', fname)
        label = match.group(1) if match else "Unknown"
        result = verify_stem(fp, label, ref_vocal, ref_instrumental, ref_mix, min_len, sr)
        results.append(result)

    return results


def print_verification_report(results):
    """Print a formatted verification report."""
    print(f"\n{'File':<60} {'Label':<15} {'Corr-Voc':>8} {'Corr-Inst':>9} {'Corr-Mix':>8} {'Content':<15} {'Match'}")
    print("-" * 130)
    for r in results:
        short = os.path.basename(r.file_path)[:57]
        status = "OK" if r.label_matches else "MISMATCH"
        print(f"{short:<60} {r.label:<15} {r.corr_vocal:>8.3f} {r.corr_instrumental:>9.3f} {r.corr_mix:>8.3f} {r.detected_content:<15} {status}")
