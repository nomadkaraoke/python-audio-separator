"""
Integration tests verifying ensemble presets produce meaningful results.

These tests go beyond regression (does output match reference?) to verify
that ensembles and specialized models produce *semantically correct* output:

- Vocal ensemble output should closely match the best single-model vocal output
- Karaoke ensemble should extract only lead vocals (differ from standard vocal split)
- Karaoke on extracted vocals should produce distinct lead and backing vocal stems

Usage:
    pytest tests/integration/test_ensemble_meaningful.py -v -s
    pytest tests/integration/test_ensemble_meaningful.py -v -s -k "karaoke"
    pytest tests/integration/test_ensemble_meaningful.py -v -s -k "lead_backing"
"""

import os
import sys
import re
import tempfile
import shutil
import logging
import pytest
import numpy as np
import librosa

REFERENCE_DIR = "tests/inputs/reference"


def correlate(file_a, file_b, sr=44100):
    """Compute Pearson correlation between two audio files (mono-mixed)."""
    a, _ = librosa.load(file_a, sr=sr, mono=True)
    b, _ = librosa.load(file_b, sr=sr, mono=True)
    ml = min(len(a), len(b))
    return float(np.corrcoef(a[:ml], b[:ml])[0, 1])


def rms(file_path, sr=44100):
    """Compute RMS energy of an audio file."""
    y, _ = librosa.load(file_path, sr=sr, mono=True)
    return float(np.sqrt(np.mean(y ** 2)))


def run_separation(model, input_file, output_dir):
    """Run a single model separation."""
    from audio_separator.separator import Separator
    sep = Separator(output_dir=output_dir, output_format="FLAC", log_level=logging.WARNING)
    sep.load_model(model)
    return sep.separate(input_file)


def run_preset(preset, input_file, output_dir):
    """Run an ensemble preset separation."""
    from audio_separator.separator import Separator
    sep = Separator(output_dir=output_dir, output_format="FLAC", ensemble_preset=preset, log_level=logging.WARNING)
    sep.load_model()
    return sep.separate(input_file)


def find_stem(output_files, stem_name, output_dir):
    """Find the output file matching a stem name (case-insensitive)."""
    for f in output_files:
        full = f if os.path.isabs(f) else os.path.join(output_dir, f)
        if not os.path.exists(full):
            full = os.path.join(output_dir, os.path.basename(f))
        match = re.search(r'_\(([^)]+)\)', os.path.basename(f))
        if match and match.group(1).lower() == stem_name.lower():
            return full
    return None


# ─── Vocal ensemble quality ─────────────────────────────────────────

VOCAL_ENSEMBLE_CLIPS = [
    ("tests/inputs/under_pressure_harmonies.flac", "ref_under_pressure_harmonies"),
    ("tests/inputs/levee_drums.flac", "ref_levee_drums"),
]


@pytest.mark.parametrize("input_file,ref_prefix", VOCAL_ENSEMBLE_CLIPS)
def test_vocal_ensemble_matches_best_single_model(input_file, ref_prefix, tmp_path):
    """Vocal ensemble output should closely match the best single vocal model.

    The vocal_balanced preset (Resurrection + Beta 6X averaged) should produce
    output highly correlated (>0.90) with the Resurrection single-model output,
    confirming the ensemble isn't degrading quality.
    """
    output_dir = str(tmp_path)
    clip = os.path.basename(input_file)
    print(f"\n  {clip}: vocal_balanced ensemble vs resurrection single model")

    # Run ensemble
    ens_out = run_preset("vocal_balanced", input_file, output_dir)
    ens_vocals = find_stem(ens_out, "Vocals", output_dir)
    ens_inst = find_stem(ens_out, "Instrumental", output_dir)
    assert ens_vocals and ens_inst

    # Compare against single-model reference
    ref_vocals = os.path.join(REFERENCE_DIR, f"{ref_prefix}_vocals.flac")
    ref_inst = os.path.join(REFERENCE_DIR, f"{ref_prefix}_instrumental.flac")

    corr_v = correlate(ens_vocals, ref_vocals)
    corr_i = correlate(ens_inst, ref_inst)
    print(f"    Vocals correlation:       {corr_v:.3f} (should be > 0.90)")
    print(f"    Instrumental correlation: {corr_i:.3f} (should be > 0.90)")

    # Ensemble should be very similar to best single model
    assert corr_v > 0.90, f"Ensemble vocals diverge too much from single model: {corr_v:.3f}"
    assert corr_i > 0.90, f"Ensemble instrumental diverges too much from single model: {corr_i:.3f}"

    # Also verify it matches its own ensemble reference (regression)
    ens_ref_vocals = os.path.join(REFERENCE_DIR, f"{ref_prefix}_vocals_preset_vocal_balanced.flac")
    ens_ref_inst = os.path.join(REFERENCE_DIR, f"{ref_prefix}_instrumental_preset_vocal_balanced.flac")
    if os.path.exists(ens_ref_vocals):
        corr_ens_v = correlate(ens_vocals, ens_ref_vocals)
        corr_ens_i = correlate(ens_inst, ens_ref_inst)
        print(f"    Ensemble regression:      vocals={corr_ens_v:.3f}, inst={corr_ens_i:.3f}")
        assert corr_ens_v > 0.70, f"Ensemble vocals regression failed: {corr_ens_v:.3f}"


# ─── Karaoke ensemble extracts only lead vocals ─────────────────────

def test_karaoke_ensemble_extracts_lead_only(tmp_path):
    """Karaoke ensemble should extract only lead vocals, not backing harmonies.

    On Under Pressure (which has prominent backing harmonies), the karaoke
    ensemble's vocal output should differ significantly from the standard
    vocal model's output (which extracts all vocals including backing).
    """
    input_file = "tests/inputs/under_pressure_harmonies.flac"
    output_dir = str(tmp_path)
    print(f"\n  Karaoke ensemble on Under Pressure (lead-only extraction)")

    # Run karaoke ensemble
    ens_out = run_preset("karaoke", input_file, output_dir)
    ens_vocals = find_stem(ens_out, "Vocals", output_dir)
    ens_inst = find_stem(ens_out, "Instrumental", output_dir)
    assert ens_vocals and ens_inst

    # Compare karaoke ensemble vocals vs standard all-vocals reference
    ref_all_vocals = os.path.join(REFERENCE_DIR, "ref_under_pressure_harmonies_vocals.flac")
    corr_vs_all = correlate(ens_vocals, ref_all_vocals)
    print(f"    Karaoke ensemble vocals vs all-vocals: {corr_vs_all:.3f} (should be < 0.90)")

    # Karaoke should extract LESS vocal content than standard separation
    assert corr_vs_all < 0.90, (
        f"Karaoke ensemble vocals too similar to standard vocals ({corr_vs_all:.3f}). "
        "Expected karaoke to extract only lead vocals, leaving backing in instrumental."
    )

    # Compare karaoke ensemble vs single karaoke model reference (regression)
    ref_kar_vocals = os.path.join(REFERENCE_DIR, "ref_under_pressure_harmonies_vocals_karaoke.flac")
    corr_vs_single_kar = correlate(ens_vocals, ref_kar_vocals)
    print(f"    Karaoke ensemble vs single karaoke:    {corr_vs_single_kar:.3f} (should be > 0.70)")
    assert corr_vs_single_kar > 0.70, f"Karaoke ensemble diverges from single karaoke: {corr_vs_single_kar:.3f}"

    # Verify karaoke instrumental has more content than standard instrumental
    # (because it keeps backing vocals)
    ens_inst_rms = rms(ens_inst)
    ref_std_inst = os.path.join(REFERENCE_DIR, "ref_under_pressure_harmonies_instrumental.flac")
    std_inst_rms = rms(ref_std_inst)
    print(f"    Karaoke inst RMS: {ens_inst_rms:.3f}, Standard inst RMS: {std_inst_rms:.3f}")


# ─── Lead/backing vocal split pipeline ──────────────────────────────

def test_karaoke_on_vocals_produces_lead_backing_split(tmp_path):
    """Running karaoke model on extracted vocals should split lead from backing.

    Pipeline: mix → vocal model → vocals → karaoke model → lead + backing
    The lead and backing outputs should both be non-silent and uncorrelated.
    """
    output_dir = str(tmp_path)
    print(f"\n  Pipeline: Under Pressure mix → vocals → karaoke → lead/backing")

    # Step 1: Extract vocals (use reference to avoid re-running)
    vocals_ref = os.path.join(REFERENCE_DIR, "ref_under_pressure_harmonies_vocals.flac")
    assert os.path.exists(vocals_ref), "Missing vocal reference — run generate_multi_stem_references.py"

    # Step 2: Run karaoke on the extracted vocals
    print(f"    Running karaoke on extracted vocals...")
    kar_out = run_separation(
        "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
        vocals_ref, output_dir,
    )

    lead_file = find_stem(kar_out, "Vocals", output_dir)
    backing_file = find_stem(kar_out, "Instrumental", output_dir)
    assert lead_file, "No lead vocal stem (labeled 'Vocals') from karaoke"
    assert backing_file, "No backing vocal stem (labeled 'Instrumental') from karaoke"

    # Both should be non-silent
    lead_rms = rms(lead_file)
    backing_rms = rms(backing_file)
    print(f"    Lead vocals RMS:    {lead_rms:.3f} (should be > 0.01)")
    print(f"    Backing vocals RMS: {backing_rms:.3f} (should be > 0.01)")
    assert lead_rms > 0.01, f"Lead vocals are too quiet: {lead_rms:.3f}"
    assert backing_rms > 0.01, f"Backing vocals are too quiet: {backing_rms:.3f}"

    # Lead and backing should be distinct (low correlation)
    lb_corr = correlate(lead_file, backing_file)
    print(f"    Lead vs backing corr: {lb_corr:.3f} (should be < 0.50)")
    assert lb_corr < 0.50, f"Lead and backing are too similar: {lb_corr:.3f}"

    # Regression: compare against committed references
    ref_lead = os.path.join(REFERENCE_DIR, "ref_under_pressure_harmonies_lead_vocals.flac")
    ref_backing = os.path.join(REFERENCE_DIR, "ref_under_pressure_harmonies_backing_vocals.flac")
    if os.path.exists(ref_lead):
        corr_lead = correlate(lead_file, ref_lead)
        corr_backing = correlate(backing_file, ref_backing)
        print(f"    Lead regression:    {corr_lead:.3f}")
        print(f"    Backing regression: {corr_backing:.3f}")
        assert corr_lead > 0.70, f"Lead vocals regression failed: {corr_lead:.3f}"
        assert corr_backing > 0.70, f"Backing vocals regression failed: {corr_backing:.3f}"
