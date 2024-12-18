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
        # Expanded stem mapping to include "no-stem" outputs
        stem_mapping = {"Vocals": "vocals", "Instrumental": "instrumental", "Drums": "drums", "Bass": "bass", "Other": "other", "No Drums": "nodrums", "No Bass": "nobass", "No Other": "noother"}

        # Perform separation if needed
        if not os.path.exists(output_dir) or not os.listdir(output_dir):
            logger.info("Performing separation...")
            separator = Separator(output_dir=output_dir)
            separator.load_model(model_filename=test_model)
            separator.separate(os.path.join(track_path, "mixture.wav"), custom_output_names=stem_mapping)

        # Check which stems were actually created and pair them appropriately
        available_stems = {}
        stem_pairs = {"drums": "nodrums", "bass": "nobass", "other": "noother", "vocals": "instrumental"}

        for main_stem, no_stem in stem_pairs.items():
            # Construct full file paths for both the isolated stem and its complement
            main_path = os.path.join(output_dir, f"{main_stem}.wav")
            no_stem_path = os.path.join(output_dir, f"{no_stem}.wav")

            # Only process this pair if both files exist
            if os.path.exists(main_path) and os.path.exists(no_stem_path):
                # Add the main stem with its path to available_stems
                available_stems[main_stem] = main_path  # This is already using the correct musdb name

                # For the complement stem, always use "accompaniment" as that's what museval expects
                available_stems["accompaniment"] = no_stem_path

        if not available_stems:
            logger.info(f"No evaluatable stems found for model {test_model}, skipping evaluation")
            return None, None

        # Get track from MUSDB
        track = next((t for t in mus_db if t.name == track_name), None)
        if track is None:
            raise ValueError(f"Track {track_name} not found in MUSDB18")

        # Load available stems
        estimates = {}
        for stem_name, stem_path in available_stems.items():
            audio, _ = sf.read(stem_path)
            if len(audio.shape) == 1:
                audio = np.expand_dims(audio, axis=1)
            estimates[stem_name] = audio

        # Evaluate using museval
        logger.info(f"Evaluating stems: {list(estimates.keys())}")
        scores = museval.eval_mus_track(track, estimates, output_dir=output_dir, mode="v4")

        # Move and rename the results file
        test_results = os.path.join(output_dir, "test", f"{track_name}.json")
        train_results = os.path.join(output_dir, "train", f"{track_name}.json")

        if os.path.exists(test_results):
            os.rename(test_results, results_file)
            os.rmdir(os.path.join(output_dir, "test"))
        elif os.path.exists(train_results):
            os.rename(train_results, results_file)
            os.rmdir(os.path.join(output_dir, "train"))

    # Calculate aggregate scores for available stems
    results_store = museval.EvalStore()
    results_store.add_track(scores.df)
    methods = museval.MethodStore()
    methods.add_evalstore(results_store, name=test_model)
    agg_scores = methods.agg_frames_tracks_scores()

    # Return the aggregate scores in a structured format with 6 significant figures
    model_results = {"track_name": track_name, "scores": {}}

    for stem in ["vocals", "drums", "bass", "other", "accompaniment"]:
        try:
            stem_scores = {metric: float(f"{agg_scores.loc[(test_model, stem, metric)]:.6g}") for metric in ["SDR", "SIR", "SAR", "ISR"]}
            # Rename 'accompaniment' to 'instrumental' in the output
            output_stem = "instrumental" if stem == "accompaniment" else stem
            model_results["scores"][output_stem] = stem_scores
        except KeyError:
            continue

    return scores, model_results if model_results["scores"] else None


def convert_decimal_to_float(obj):
    """Recursively converts Decimal objects to floats in a nested structure."""
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_decimal_to_float(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_decimal_to_float(x) for x in obj]
    return obj


def calculate_median_scores(track_scores):
    """Calculate median scores across all tracks for each stem and metric"""
    # Initialize containers for each stem's metrics
    stem_metrics = {
        "vocals": {"SDR": [], "SIR": [], "SAR": [], "ISR": []},
        "drums": {"SDR": [], "SIR": [], "SAR": [], "ISR": []},
        "bass": {"SDR": [], "SIR": [], "SAR": [], "ISR": []},
        "instrumental": {"SDR": [], "SIR": [], "SAR": [], "ISR": []},
    }

    # Collect all scores for each stem and metric
    for track_score in track_scores:
        if track_score is not None and "scores" in track_score:
            for stem, metrics in track_score["scores"].items():
                if stem in stem_metrics:
                    for metric, value in metrics.items():
                        stem_metrics[stem][metric].append(value)

    # Calculate medians for each stem and metric
    median_scores = {}
    for stem, metrics in stem_metrics.items():
        if any(metrics.values()):  # Only include stems that have scores
            median_scores[stem] = {metric: float(f"{np.median(values):.6g}") for metric, values in metrics.items() if values}  # Only include metrics that have values

    return median_scores


def main():
    logger.info("Starting model evaluation script...")
    os.makedirs(RESULTS_PATH, exist_ok=True)

    # Load existing results if available
    combined_results_path = "audio_separator/models-scores.json"
    combined_results = {}
    if os.path.exists(combined_results_path):
        logger.info("Loading existing combined results...")
        with open(combined_results_path) as f:
            combined_results = json.load(f)

    # Define known demucs model stems
    DEMUCS_STEMS = {
        "htdemucs.yaml": {"instruments": ["vocals", "drums", "bass", "other"], "target_instrument": None},
        "htdemucs_ft.yaml": {"instruments": ["vocals", "drums", "bass", "other"], "target_instrument": None},
        "hdemucs_mmi.yaml": {"instruments": ["vocals", "drums", "bass", "other"], "target_instrument": None},
        "htdemucs_6s.yaml": {"instruments": ["vocals", "drums", "bass", "guitar", "piano", "other"], "target_instrument": None},
    }

    # Get list of all available models
    logger.info("Getting list of available models...")
    separator = Separator()
    models_by_type = separator.list_supported_model_files()

    # Iterate through models and load each one
    for model_type, models in models_by_type.items():
        logger.info(f"\nProcessing model type: {model_type}")
        for model_name, model_info in models.items():
            test_model = model_info.get("filename")
            if not test_model:
                logger.warning(f"No filename found for model {model_name}, skipping...")
                continue

            logger.info(f"\n=== Analyzing model: {model_name} (filename: {test_model}) ===")
            try:
                separator.load_model(model_filename=test_model)
                model_data = separator.model_instance.model_data
                logger.info(f"Raw model_data: {json.dumps(model_data, indent=2)}")

                # Initialize model entry if it doesn't exist
                if test_model not in combined_results:
                    logger.info(f"Initializing new entry for {test_model}")
                    combined_results[test_model] = {"model_name": model_name, "track_scores": [], "median_scores": {}, "stems": [], "target_stem": None}

                # Handle demucs models specially
                if test_model in DEMUCS_STEMS:
                    logger.info(f"Processing as Demucs model: {test_model}")
                    logger.info(f"Demucs config: {DEMUCS_STEMS[test_model]}")
                    combined_results[test_model]["stems"] = [s.lower() for s in DEMUCS_STEMS[test_model]["instruments"]]
                    combined_results[test_model]["target_stem"] = DEMUCS_STEMS[test_model]["target_instrument"].lower() if DEMUCS_STEMS[test_model]["target_instrument"] else None
                    logger.info(f"Set stems to: {combined_results[test_model]['stems']}")
                    logger.info(f"Set target_stem to: {combined_results[test_model]['target_stem']}")

                # Extract stem information for other models
                elif "training" in model_data:
                    logger.info("Processing model with training data")
                    instruments = model_data["training"].get("instruments", [])
                    target = model_data["training"].get("target_instrument")
                    logger.info(f"Found instruments: {instruments}")
                    logger.info(f"Found target: {target}")
                    combined_results[test_model]["stems"] = [s.lower() for s in instruments] if instruments else []
                    combined_results[test_model]["target_stem"] = target.lower() if target else None
                    logger.info(f"Set stems to: {combined_results[test_model]['stems']}")
                    logger.info(f"Set target_stem to: {combined_results[test_model]['target_stem']}")

                elif "primary_stem" in model_data:
                    logger.info("Processing model with primary_stem")
                    primary_stem = model_data["primary_stem"].lower()
                    logger.info(f"Found primary_stem: {primary_stem}")

                    if primary_stem == "vocals":
                        other_stem = "instrumental"
                    elif primary_stem == "instrumental":
                        other_stem = "vocals"
                    else:
                        if primary_stem.startswith("no "):
                            other_stem = primary_stem[3:]  # Remove "no " prefix
                        else:
                            other_stem = "no " + primary_stem
                    logger.info(f"Determined other_stem: {other_stem}")

                    instruments = [primary_stem, other_stem]
                    combined_results[test_model]["stems"] = instruments
                    combined_results[test_model]["target_stem"] = primary_stem
                    logger.info(f"Set stems to: {combined_results[test_model]['stems']}")
                    logger.info(f"Set target_stem to: {combined_results[test_model]['target_stem']}")

                else:
                    logger.warning(f"No recognized stem information found in model data for {test_model}")
                    combined_results[test_model]["stems"] = []
                    combined_results[test_model]["target_stem"] = None

                logger.info(f"Final model configuration for {test_model}:")
                logger.info(f"Stems: {combined_results[test_model]['stems']}")
                logger.info(f"Target stem: {combined_results[test_model]['target_stem']}")

            except Exception as e:
                logger.error(f"Error loading model {test_model}: {str(e)}")
                logger.exception("Full exception details:")
                continue

    # Save the combined results after model inspection
    logger.info("Saving model stem information...")
    os.makedirs(os.path.dirname(combined_results_path), exist_ok=True)
    with open(combined_results_path, "w") as f:
        json.dump(combined_results, f, indent=2)

    logger.info("Model stem information saved")

    # Initialize MUSDB once at the start
    logger.info("Initializing MUSDB database...")
    mus = musdb.DB(root=MUSDB_PATH, is_wav=True)

    # Process all tracks in MUSDB18
    for track in mus.tracks:
        track_name = track.name
        track_path = os.path.dirname(track.path)
        logger.info(f"Processing track: {track_name}")

        # Process all models for this track
        for model_type, models in models_by_type.items():
            for model_name, model_info in models.items():
                # Get the filename from the model_info dictionary
                test_model = model_info.get("filename")

                # Skip if no filename is found
                if not test_model:
                    logger.warning(f"No filename found for model {model_name}, skipping...")
                    continue

                # Initialize model entry if it doesn't exist
                if test_model not in combined_results:
                    combined_results[test_model] = {"model_name": model_name, "track_scores": [], "median_scores": {}, "stems": [], "target_stem": None}

                # Handle demucs models specially
                if test_model in DEMUCS_STEMS:
                    combined_results[test_model]["stems"] = [s.lower() for s in DEMUCS_STEMS[test_model]["instruments"]]
                    combined_results[test_model]["target_stem"] = DEMUCS_STEMS[test_model]["target_instrument"].lower() if DEMUCS_STEMS[test_model]["target_instrument"] else None

                # Extract stem information for other models
                elif "training" in model_data:
                    instruments = model_data["training"].get("instruments", [])
                    target = model_data["training"].get("target_instrument")
                    combined_results[test_model]["stems"] = [s.lower() for s in instruments] if instruments else []
                    combined_results[test_model]["target_stem"] = target.lower() if target else None

                elif "primary_stem" in model_data:
                    primary_stem = model_data["primary_stem"].lower()
                    if primary_stem == "vocals":
                        other_stem = "instrumental"
                    elif primary_stem == "instrumental":
                        other_stem = "vocals"
                    else:
                        other_stem = "no " + primary_stem

                    instruments = [primary_stem, other_stem]
                    combined_results[test_model]["stems"] = instruments
                    combined_results[test_model]["target_stem"] = primary_stem

                else:
                    combined_results[test_model]["stems"] = []
                    combined_results[test_model]["target_stem"] = None
                    logger.info("No stem information found in model data")

                # Check if track already evaluated
                track_already_evaluated = any(track_score["track_name"] == track_name for track_score in combined_results[test_model]["track_scores"] if track_score is not None)

                if track_already_evaluated:
                    logger.info(f"Skipping already evaluated track {track_name} for model: {test_model}")
                    continue

                # Process the model
                logger.info(f"Processing model: {test_model}")
                try:
                    _, model_results = evaluate_track(track_name, track_path, test_model, mus)
                    if model_results:
                        combined_results[test_model]["track_scores"].append(model_results)
                    else:
                        logger.info(f"Skipping model {test_model} for track {track_name} due to no evaluatable stems")
                except Exception as e:
                    logger.error(f"Error evaluating model {test_model} with track {track_name}: {str(e)}")
                    logger.exception(f"Exception details: ", exc_info=e)
                    continue

                # Update and save results
                if combined_results[test_model]["track_scores"]:
                    median_scores = calculate_median_scores(combined_results[test_model]["track_scores"])
                    combined_results[test_model]["median_scores"] = median_scores
                else:
                    del combined_results[test_model]

                # Save results after each model
                os.makedirs(os.path.dirname(combined_results_path), exist_ok=True)
                with open(combined_results_path, "w") as f:
                    json.dump(combined_results, f, indent=2)

                if not track_already_evaluated:
                    logger.info(f"Updated combined results file with {test_model} - {track_name}")

    logger.info("Evaluation complete")
    return 0


if __name__ == "__main__":
    exit(main())
