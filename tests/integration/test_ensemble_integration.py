"""
Integration tests for ensemble preset separations.

Tests that each ensemble preset produces correct output by:
1. Running the preset on the test input audio
2. Verifying output stems contain the expected content (vocal vs instrumental)
3. Comparing output spectrograms against committed reference images
"""

import os
import sys
import tempfile
import shutil
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import generate_reference_images, compare_images
from utils_audio_verification import load_references, verify_separation_outputs


@pytest.fixture(name="input_file")
def fixture_input_file():
    """Fixture providing the test input audio file path."""
    return "tests/inputs/mardy20s.flac"


@pytest.fixture(name="reference_dir")
def fixture_reference_dir():
    """Fixture providing the reference images directory path."""
    return "tests/inputs/reference"


@pytest.fixture(name="temp_output_dir")
def fixture_temp_output_dir():
    """Fixture providing a temporary directory for output files."""
    temp_dir = tempfile.mkdtemp(prefix="ensemble-test-")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(name="audio_references")
def fixture_audio_references():
    """Fixture providing loaded audio references for content verification."""
    ref_vocal, ref_inst, ref_mix, min_len = load_references()
    return ref_vocal, ref_inst, ref_mix, min_len


def validate_audio_output(output_file, reference_dir, waveform_threshold=0.90, spectrogram_threshold=0.80):
    """Validate an audio output file by comparing its waveform and spectrogram with reference images."""
    temp_dir = os.path.join(os.path.dirname(output_file), "temp_images")
    os.makedirs(temp_dir, exist_ok=True)

    output_filename = os.path.basename(output_file)
    name_without_ext = os.path.splitext(output_filename)[0]

    actual_waveform_path, actual_spectrogram_path = generate_reference_images(
        output_file, temp_dir, prefix="actual_"
    )

    expected_waveform_path = os.path.join(reference_dir, f"expected_{name_without_ext}_waveform.png")
    expected_spectrogram_path = os.path.join(reference_dir, f"expected_{name_without_ext}_spectrogram.png")

    if not os.path.exists(expected_waveform_path) or not os.path.exists(expected_spectrogram_path):
        print(f"Warning: Reference images not found for {output_file}")
        print(f"Expected: {expected_waveform_path}")
        print(f"Run generate_reference_images_ensemble.py to create them")
        return False, False

    waveform_similarity, waveform_match = compare_images(
        expected_waveform_path, actual_waveform_path,
        min_similarity_threshold=waveform_threshold
    )

    spectrogram_similarity, spectrogram_match = compare_images(
        expected_spectrogram_path, actual_spectrogram_path,
        min_similarity_threshold=spectrogram_threshold
    )

    print(f"  Waveform SSIM: {waveform_similarity:.4f} (threshold: {waveform_threshold:.2f}, match: {waveform_match})")
    print(f"  Spectrogram SSIM: {spectrogram_similarity:.4f} (threshold: {spectrogram_threshold:.2f}, match: {spectrogram_match})")

    return waveform_match, spectrogram_match


# Similarity thresholds — ensemble outputs can vary slightly across runs
DEFAULT_THRESHOLDS = (0.90, 0.80)  # (waveform, spectrogram)

# All 9 ensemble presets with their expected output stems
ENSEMBLE_PRESET_PARAMS = [
    ("instrumental_clean", [
        "mardy20s_(Vocals)_preset_instrumental_clean.flac",
        "mardy20s_(Instrumental)_preset_instrumental_clean.flac",
    ]),
    ("instrumental_full", [
        "mardy20s_(Vocals)_preset_instrumental_full.flac",
        "mardy20s_(Instrumental)_preset_instrumental_full.flac",
    ]),
    ("instrumental_balanced", [
        "mardy20s_(Vocals)_preset_instrumental_balanced.flac",
        "mardy20s_(Instrumental)_preset_instrumental_balanced.flac",
    ]),
    ("instrumental_low_resource", [
        "mardy20s_(Vocals)_preset_instrumental_low_resource.flac",
        "mardy20s_(Instrumental)_preset_instrumental_low_resource.flac",
    ]),
    ("vocal_balanced", [
        "mardy20s_(Vocals)_preset_vocal_balanced.flac",
        "mardy20s_(Instrumental)_preset_vocal_balanced.flac",
    ]),
    ("vocal_clean", [
        "mardy20s_(Vocals)_preset_vocal_clean.flac",
        "mardy20s_(Instrumental)_preset_vocal_clean.flac",
    ]),
    ("vocal_full", [
        "mardy20s_(Vocals)_preset_vocal_full.flac",
        "mardy20s_(Instrumental)_preset_vocal_full.flac",
    ]),
    ("vocal_rvc", [
        "mardy20s_(Vocals)_preset_vocal_rvc.flac",
        "mardy20s_(Instrumental)_preset_vocal_rvc.flac",
    ]),
    ("karaoke", [
        "mardy20s_(Vocals)_preset_karaoke.flac",
        "mardy20s_(Instrumental)_preset_karaoke.flac",
    ]),
]


@pytest.mark.parametrize("preset,expected_files", ENSEMBLE_PRESET_PARAMS)
def test_ensemble_preset(preset, expected_files, input_file, reference_dir, temp_output_dir, audio_references):
    """Test that an ensemble preset produces correctly labeled and spectrogram-matching output."""
    from audio_separator.separator import Separator

    print(f"\n{'='*60}")
    print(f"  Testing preset: {preset}")
    print(f"{'='*60}")

    # Run separation
    separator = Separator(
        output_dir=temp_output_dir,
        output_format="FLAC",
        ensemble_preset=preset,
    )
    separator.load_model()
    output_files = separator.separate(input_file)

    # Check expected files were created
    output_basenames = [os.path.basename(f) for f in output_files]
    for expected in expected_files:
        assert expected in output_basenames, (
            f"Expected output '{expected}' not found. Got: {output_basenames}"
        )

    # Check files exist and are non-empty
    for output_file in output_files:
        full_path = output_file if os.path.isabs(output_file) else os.path.join(temp_output_dir, output_file)
        assert os.path.exists(full_path), f"Output file does not exist: {full_path}"
        assert os.path.getsize(full_path) > 0, f"Output file is empty: {full_path}"

    # Content verification — ensure stems contain what their labels claim
    ref_vocal, ref_inst, ref_mix, min_len = audio_references
    full_paths = [
        f if os.path.isabs(f) else os.path.join(temp_output_dir, f)
        for f in output_files
    ]
    verifications = verify_separation_outputs(full_paths, ref_vocal, ref_inst, ref_mix, min_len)

    for v in verifications:
        print(f"  {v.label:<15} → {v.detected_content:<15} corr_v={v.corr_vocal:.3f} corr_i={v.corr_instrumental:.3f}")
        assert v.label_matches, (
            f"Stem '{v.label}' contains {v.detected_content} "
            f"(corr_vocal={v.corr_vocal:.3f}, corr_inst={v.corr_instrumental:.3f})"
        )

    # Spectrogram comparison against reference images
    if os.environ.get("SKIP_AUDIO_VALIDATION") == "1":
        print("  Skipping spectrogram validation (SKIP_AUDIO_VALIDATION=1)")
        return

    waveform_threshold, spectrogram_threshold = DEFAULT_THRESHOLDS

    for output_file in output_files:
        full_path = output_file if os.path.isabs(output_file) else os.path.join(temp_output_dir, output_file)
        print(f"\n  Validating: {os.path.basename(output_file)}")
        waveform_match, spectrogram_match = validate_audio_output(
            full_path, reference_dir,
            waveform_threshold=waveform_threshold,
            spectrogram_threshold=spectrogram_threshold,
        )

        assert waveform_match, f"Waveform mismatch for {output_file}"
        assert spectrogram_match, f"Spectrogram mismatch for {output_file}"

    print(f"\n  Preset '{preset}' passed all checks")
