#!/usr/bin/env python
import os
import museval
import numpy as np
import soundfile as sf
from audio_separator.separator import Separator
from pathlib import Path
import json
import logging
import musdb
from decimal import Decimal
import pandas as pd


# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MUSDB_PATH = "tests/model-metrics/datasets/musdb18hq"
RESULTS_PATH = "tests/model-metrics/results"


def evaluate_track(track_name, track_path, test_model, mus_db):
    """Evaluate a single track using a specific model"""
    logger.info(f"Evaluating track: {track_name} with model: {test_model}")

    # Set output directory for this separation
    output_dir = os.path.join(RESULTS_PATH, test_model, track_name)
    os.makedirs(output_dir, exist_ok=True)

    # Check if evaluation results already exist
    results_file = os.path.join(output_dir, "museval-results.json")
    if os.path.exists(results_file):
        logger.info("Found existing evaluation results, loading from file...")
        with open(results_file) as f:
            json_data = json.load(f)
        scores = museval.TrackStore(track_name)
        scores.scores = json_data
    else:
        # Check if separated files already exist
        vocals_path = os.path.join(output_dir, "vocals.wav")
        instrumental_path = os.path.join(output_dir, "instrumental.wav")

        if not (os.path.exists(vocals_path) and os.path.exists(instrumental_path)):
            logger.info("Performing separation...")
            separator = Separator(output_dir=output_dir)
            separator.load_model(model_filename=test_model)
            separator.separate(os.path.join(track_path, "mixture.wav"), custom_output_names={"Vocals": "vocals", "Instrumental": "instrumental"})

        # Get track from MUSDB
        track = next((t for t in mus_db if t.name == track_name), None)
        if track is None:
            raise ValueError(f"Track {track_name} not found in MUSDB18")

        # Load estimated stems
        estimates = {}
        for stem_name in ["vocals", "accompaniment"]:
            stem_path = vocals_path if stem_name == "vocals" else instrumental_path
            audio, _ = sf.read(stem_path)
            if len(audio.shape) == 1:
                audio = np.expand_dims(audio, axis=1)
            estimates[stem_name] = audio

        # Evaluate using museval
        logger.info("Performing evaluation...")
        scores = museval.eval_mus_track(track, estimates, output_dir=output_dir, mode="v4")

        # Move and rename the results file
        test_results = os.path.join(output_dir, "test", f"{track_name}.json")
        if os.path.exists(test_results):
            os.rename(test_results, results_file)
            os.rmdir(os.path.join(output_dir, "test"))

    # Calculate aggregate scores
    results_store = museval.EvalStore()
    results_store.add_track(scores.df)
    methods = museval.MethodStore()
    methods.add_evalstore(results_store, name=test_model)
    agg_scores = methods.agg_frames_tracks_scores()

    # Log results
    logger.info(
        "Vocals (SDR, SIR, SAR, ISR): %.2f, %.2f, %.2f, %.2f",
        agg_scores.loc[(test_model, "vocals", "SDR")],
        agg_scores.loc[(test_model, "vocals", "SIR")],
        agg_scores.loc[(test_model, "vocals", "SAR")],
        agg_scores.loc[(test_model, "vocals", "ISR")],
    )

    logger.info(
        "Accompaniment (SDR, SIR, SAR, ISR): %.2f, %.2f, %.2f, %.2f",
        agg_scores.loc[(test_model, "accompaniment", "SDR")],
        agg_scores.loc[(test_model, "accompaniment", "SIR")],
        agg_scores.loc[(test_model, "accompaniment", "SAR")],
        agg_scores.loc[(test_model, "accompaniment", "ISR")],
    )


def convert_decimal_to_float(obj):
    """Recursively converts Decimal objects to floats in a nested structure."""
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_decimal_to_float(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_decimal_to_float(x) for x in obj]
    return obj


def main():
    logger.info("Starting model evaluation script...")

    # Create results directory
    os.makedirs(RESULTS_PATH, exist_ok=True)

    # Initialize MUSDB once at the start
    logger.info("Initializing MUSDB database...")
    mus = musdb.DB(root=MUSDB_PATH, is_wav=True)

    # Get list of all available models
    logger.info("Getting list of available models...")
    separator = Separator()
    models_by_type = separator.list_supported_model_files()

    # Just get the first model for testing
    test_model = None
    for model_type, models in models_by_type.items():
        for model_name, model_info in models.items():
            if isinstance(model_info, str):
                test_model = model_info
                break
            elif isinstance(model_info, dict):
                for file_name in model_info.keys():
                    if file_name.endswith((".onnx", ".pth", ".ckpt")):
                        test_model = file_name
                        break
        if test_model:
            break

    logger.info(f"Selected model for testing: {test_model}")

    # Get first track from MUSDB18
    logger.info("Looking for first track in MUSDB18 dataset...")
    first_track = next(Path(MUSDB_PATH).glob("**/mixture.wav"))
    track_path = first_track.parent
    track_name = track_path.name
    logger.info(f"Found track: {track_name} at path: {track_path}")

    try:
        logger.info("Starting track evaluation...")
        evaluate_track(track_name, track_path, test_model, mus)
        logger.info("Completed evaluation")

    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        logger.error("Error details:", exc_info=True)
        return 1

    # Clean up any test directories that might have been created
    test_dir = os.path.join(RESULTS_PATH, test_model, track_name, "test")
    if os.path.exists(test_dir):
        import shutil

        shutil.rmtree(test_dir)

    return 0


if __name__ == "__main__":
    exit(main())
