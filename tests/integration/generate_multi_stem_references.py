#!/usr/bin/env python3
"""
Generate reference stems for multi-stem integration tests.

Runs best-in-class models on test clips to produce reference stems that
other models' outputs can be compared against.

Usage:
    python tests/integration/generate_multi_stem_references.py

Pipelines:
    1. All 4 clips → bs_roformer_vocals_resurrection_unwa → vocals + instrumental refs
    2. levee_drums, clocks_piano → htdemucs_ft → drums/bass/other/vocals refs
    3. levee_drums → htdemucs_ft drums stem → MDX23C-DrumSep → kit part refs
    4. levee_drums, clocks_piano → karaoke model → karaoke vocal/instrumental refs
    5. sing_sing_sing_brass → 17_HP-Wind_Inst-UVR → woodwind refs
    6. only_time_reverb → resurrection vocals → dereverb → noreverb/reverb refs
"""

import os
import sys
import shutil
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_separator.separator import Separator

INPUTS_DIR = "tests/inputs"
REFERENCE_DIR = "tests/inputs/reference"

CLIPS = {
    "levee_drums": f"{INPUTS_DIR}/levee_drums.flac",
    "clocks_piano": f"{INPUTS_DIR}/clocks_piano.flac",
    "sing_sing_sing_brass": f"{INPUTS_DIR}/sing_sing_sing_brass.flac",
    "only_time_reverb": f"{INPUTS_DIR}/only_time_reverb.flac",
}


def run_model(model, input_file, output_dir):
    """Run a model and return output file paths."""
    sep = Separator(output_dir=output_dir, output_format="FLAC")
    sep.load_model(model)
    return sep.separate(input_file)


def find_stem_file(output_files, stem_name, output_dir):
    """Find output file matching stem name (uses last parenthesized group for pipeline support)."""
    import re
    for f in output_files:
        full = f if os.path.isabs(f) else os.path.join(output_dir, f)
        if not os.path.exists(full):
            full = os.path.join(output_dir, os.path.basename(f))
        matches = re.findall(r'_\(([^)]+)\)', os.path.basename(f))
        if matches and matches[-1].lower() == stem_name.lower():
            return full
    return None


def main():
    os.makedirs(REFERENCE_DIR, exist_ok=True)
    temp_dir = tempfile.mkdtemp(prefix="multi-stem-refs-")

    try:
        # 1. Vocal/Instrumental on all clips
        print("=== Vocal/Instrumental references (resurrection) ===")
        model = "bs_roformer_vocals_resurrection_unwa.ckpt"
        for clip_name, clip_path in CLIPS.items():
            print(f"  {clip_name}...")
            outputs = run_model(model, clip_path, temp_dir)
            vocals = find_stem_file(outputs, "vocals", temp_dir)
            inst = find_stem_file(outputs, "other", temp_dir)
            shutil.copy2(vocals, f"{REFERENCE_DIR}/ref_{clip_name}_vocals.flac")
            shutil.copy2(inst, f"{REFERENCE_DIR}/ref_{clip_name}_instrumental.flac")

        # 2. htdemucs_ft 4-stem on levee + clocks
        print("\n=== 4-stem references (htdemucs_ft) ===")
        model = "htdemucs_ft.yaml"
        for clip_name in ["levee_drums", "clocks_piano"]:
            print(f"  {clip_name}...")
            outputs = run_model(model, CLIPS[clip_name], temp_dir)
            for stem in ["Vocals", "Drums", "Bass", "Other"]:
                stem_file = find_stem_file(outputs, stem, temp_dir)
                shutil.copy2(stem_file, f"{REFERENCE_DIR}/ref_{clip_name}_{stem.lower()}_htdemucs_ft.flac")

        # 3. DrumSep pipeline: levee drums stem → kit parts
        print("\n=== DrumSep pipeline references ===")
        drums_stem = find_stem_file(
            run_model("htdemucs_ft.yaml", CLIPS["levee_drums"], temp_dir),
            "Drums", temp_dir
        )
        outputs = run_model("MDX23C-DrumSep-aufr33-jarredou.ckpt", drums_stem, temp_dir)
        for stem in ["kick", "snare", "toms", "hh", "ride", "crash"]:
            stem_file = find_stem_file(outputs, stem, temp_dir)
            shutil.copy2(stem_file, f"{REFERENCE_DIR}/ref_levee_drums_{stem}_drumsep.flac")
            print(f"  {stem}: done")

        # 4. Karaoke on levee + clocks
        print("\n=== Karaoke references ===")
        model = "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"
        for clip_name in ["levee_drums", "clocks_piano"]:
            print(f"  {clip_name}...")
            outputs = run_model(model, CLIPS[clip_name], temp_dir)
            vocals = find_stem_file(outputs, "Vocals", temp_dir)
            inst = find_stem_file(outputs, "Instrumental", temp_dir)
            shutil.copy2(vocals, f"{REFERENCE_DIR}/ref_{clip_name}_vocals_karaoke.flac")
            shutil.copy2(inst, f"{REFERENCE_DIR}/ref_{clip_name}_instrumental_karaoke.flac")

        # 5. Wind/Brass
        print("\n=== Wind instrument references ===")
        outputs = run_model("17_HP-Wind_Inst-UVR.pth", CLIPS["sing_sing_sing_brass"], temp_dir)
        ww = find_stem_file(outputs, "Woodwinds", temp_dir)
        no_ww = find_stem_file(outputs, "No Woodwinds", temp_dir)
        shutil.copy2(ww, f"{REFERENCE_DIR}/ref_sing_sing_sing_brass_woodwinds.flac")
        shutil.copy2(no_ww, f"{REFERENCE_DIR}/ref_sing_sing_sing_brass_no_woodwinds.flac")

        # 6. Dereverb pipeline: only_time vocals → dereverb
        print("\n=== Dereverb pipeline references ===")
        vocals_file = f"{REFERENCE_DIR}/ref_only_time_reverb_vocals.flac"  # Already generated in step 1
        outputs = run_model("dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt", vocals_file, temp_dir)
        noreverb = find_stem_file(outputs, "noreverb", temp_dir)
        reverb = find_stem_file(outputs, "reverb", temp_dir)
        shutil.copy2(noreverb, f"{REFERENCE_DIR}/ref_only_time_reverb_vocals_noreverb.flac")
        shutil.copy2(reverb, f"{REFERENCE_DIR}/ref_only_time_reverb_vocals_reverb.flac")

        print(f"\nDone! Generated {len([f for f in os.listdir(REFERENCE_DIR) if f.startswith('ref_')])} reference stems.")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
