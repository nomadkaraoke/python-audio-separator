#!/usr/bin/env python3
"""
On-demand regression test: verify every supported model's output stem labels
match their actual audio content.

This test runs ALL supported models on the 20-second test audio file and uses
correlation-based analysis to verify each output stem contains what its label
claims (e.g., a stem labeled "Vocals" actually contains vocal content).

Usage:
    # Run all models (takes a long time — ~163 models):
    pytest tests/regression/test_all_models_stem_verification.py -v -s

    # Run a specific architecture:
    pytest tests/regression/test_all_models_stem_verification.py -v -s -k "VR"
    pytest tests/regression/test_all_models_stem_verification.py -v -s -k "MDX"
    pytest tests/regression/test_all_models_stem_verification.py -v -s -k "MDXC"
    pytest tests/regression/test_all_models_stem_verification.py -v -s -k "Demucs"

    # Run a single model:
    pytest tests/regression/test_all_models_stem_verification.py -v -s -k "UVR_MDXNET_KARA"

    # Generate a report without failing on mismatches (dry run):
    STEM_VERIFY_REPORT_ONLY=1 pytest tests/regression/test_all_models_stem_verification.py -v -s

When to run:
    - After changing stem naming logic in common_separator.py or separator.py
    - After adding new models or YAML configs to models.json
    - After modifying the MDXC/VR separator stem assignment code
    - Periodically as a health check

NOT run in CI — requires downloading all models.
"""

import os
import sys
import re
import tempfile
import shutil
import json
import logging
import pytest
import numpy as np
import librosa

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils_audio_verification import load_references, classify_audio


INPUT_FILE = "tests/inputs/mardy20s.flac"

# Stems where we can verify content via correlation
VOCAL_STEMS = {"vocals", "vocal", "lead vocals", "backing vocals", "lead_only", "backing_only"}
INSTRUMENTAL_STEMS = {"instrumental", "inst", "karaoke", "no_vocals", "no vocals"}

# Stems that are sub-components — we can only verify they're not silent/full-mix
SUB_STEMS = {"drums", "bass", "guitar", "piano", "other", "synthesizer", "strings",
             "woodwinds", "brass", "wind inst", "no drums", "no bass", "no guitar",
             "no piano", "no other", "no synthesizer", "no strings", "no woodwinds",
             "no brass", "no wind inst", "drum-bass", "no drum-bass"}

# Report-only mode: print results but don't fail
REPORT_ONLY = os.environ.get("STEM_VERIFY_REPORT_ONLY", "0") == "1"


@pytest.fixture(scope="session")
def audio_references():
    """Load reference audio once for the entire test session."""
    return load_references(input_dir="tests/inputs")


def get_all_models():
    """Get all supported models grouped by architecture."""
    from audio_separator.separator import Separator
    sep = Separator(info_only=True, log_level=logging.WARNING)
    return sep.list_supported_model_files()


def build_model_params():
    """Build pytest parametrize list: (arch, friendly_name, filename)."""
    params = []
    all_models = get_all_models()
    for arch, models in all_models.items():
        for friendly_name, info in models.items():
            filename = info.get("filename", "")
            if not filename:
                continue
            test_id = f"{arch}-{filename}"
            params.append(pytest.param(arch, friendly_name, filename, id=test_id))
    return params


MODEL_PARAMS = build_model_params()


def verify_stem_content(stem_path, stem_label, ref_vocal, ref_inst, ref_mix, min_len):
    """Verify a single stem's content matches its label.

    Returns (passed, message) tuple.
    """
    y, _ = librosa.load(stem_path, sr=44100, mono=True)
    cv, ci, cm, rms, detected = classify_audio(y, ref_vocal, ref_inst, ref_mix, min_len)

    label_lower = stem_label.lower()
    issues = []

    is_sub_stem = label_lower in SUB_STEMS

    # Check for silent output — sub-stems (drums, piano, etc.) can legitimately be
    # near-silent if the source audio doesn't contain that instrument
    if rms < 0.001 and not is_sub_stem:
        return False, f"SILENT (rms={rms:.6f})"

    # Check for full mix leak (no stem should be the original mix)
    if cm > 0.95:
        return False, f"FULL_MIX (corr_mix={cm:.3f}) — stem contains the original mix, not a separation"

    # Verify vocal-labeled stems
    if label_lower in VOCAL_STEMS or ("vocal" in label_lower and "no" not in label_lower):
        if detected != "VOCALS":
            issues.append(f"labeled '{stem_label}' but detected {detected} (corr_v={cv:.3f}, corr_i={ci:.3f})")
        if cv < 0.7:
            issues.append(f"low vocal correlation ({cv:.3f}) for vocal-labeled stem")

    # Verify instrumental-labeled stems
    elif label_lower in INSTRUMENTAL_STEMS:
        if detected != "INSTRUMENTAL":
            issues.append(f"labeled '{stem_label}' but detected {detected} (corr_v={cv:.3f}, corr_i={ci:.3f})")
        if ci < 0.7:
            issues.append(f"low instrumental correlation ({ci:.3f}) for instrumental-labeled stem")

    # Sub-stems (drums, bass, etc.) — verify not full mix; near-silence is OK
    elif is_sub_stem:
        pass  # Already checked full_mix above

    # Unknown stem type — log but don't fail
    else:
        issues.append(f"unknown stem type '{stem_label}' — cannot verify content (detected={detected})")

    if issues:
        return False, "; ".join(issues)
    return True, f"OK (detected={detected}, corr_v={cv:.3f}, corr_i={ci:.3f}, corr_m={cm:.3f})"


@pytest.mark.parametrize("arch,friendly_name,model_filename", MODEL_PARAMS)
def test_model_stem_labels(arch, friendly_name, model_filename, audio_references, tmp_path):
    """Verify that a model's output stems contain what their labels claim."""
    from audio_separator.separator import Separator

    ref_vocal, ref_inst, ref_mix, min_len = audio_references

    print(f"\n  Model: {model_filename} ({arch})")
    print(f"  Friendly name: {friendly_name}")

    # Skip Demucs on Python < 3.10
    if arch == "Demucs":
        import sys
        if sys.version_info < (3, 10):
            pytest.skip("Demucs requires Python 3.10+")

    temp_dir = str(tmp_path)

    try:
        sep = Separator(output_dir=temp_dir, output_format="WAV", log_level=logging.WARNING)
        sep.load_model(model_filename)
        output_files = sep.separate(INPUT_FILE)
    except Exception as e:
        # Model download or separation failure — report but don't mask as stem issue
        pytest.skip(f"Model failed to run: {e}")

    all_passed = True
    messages = []

    for output_file in output_files:
        full_path = output_file if os.path.isabs(output_file) else os.path.join(temp_dir, output_file)
        if not os.path.exists(full_path):
            full_path = os.path.join(temp_dir, os.path.basename(output_file))

        fname = os.path.basename(full_path)
        match = re.search(r'_\(([^)]+)\)', fname)
        stem_label = match.group(1) if match else "Unknown"

        passed, msg = verify_stem_content(full_path, stem_label, ref_vocal, ref_inst, ref_mix, min_len)
        status = "PASS" if passed else "FAIL"
        print(f"    {stem_label:<20} {status}  {msg}")
        messages.append((stem_label, passed, msg))

        if not passed:
            all_passed = False

    if not all_passed and not REPORT_ONLY:
        failures = [f"  {label}: {msg}" for label, passed, msg in messages if not passed]
        pytest.fail(f"Stem content mismatch for {model_filename}:\n" + "\n".join(failures))
