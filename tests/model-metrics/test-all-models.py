#!/usr/bin/env python
import os
import museval
import numpy as np
import soundfile as sf
from audio_separator.separator import Separator
import json
import logging
import musdb
from decimal import Decimal
import tempfile
import argparse


# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MUSDB_PATH = "/Volumes/Nomad4TBOne/python-audio-separator/tests/model-metrics/datasets/musdb18hq"
RESULTS_PATH = "/Volumes/Nomad4TBOne/python-audio-separator/tests/model-metrics/results"
COMBINED_RESULTS_PATH = "/Users/andrew/Projects/python-audio-separator/audio_separator/models-scores.json"
COMBINED_MUSEVAL_RESULTS_PATH = "/Volumes/Nomad4TBOne/python-audio-separator/tests/model-metrics/results/combined-museval-results.json"


def load_combined_results():
    """Load the combined museval results file"""
    if os.path.exists(COMBINED_MUSEVAL_RESULTS_PATH):
        logger.info("Loading combined museval results...")
        try:
            with open(COMBINED_MUSEVAL_RESULTS_PATH, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading combined results: {str(e)}")
            return {}
    else:
        logger.info("No combined results file found, creating new one")
        return {}


def save_combined_results(combined_results):
    """Save the combined museval results file"""
    logger.info("Saving combined museval results...")
    try:
        with open(COMBINED_MUSEVAL_RESULTS_PATH, "w") as f:
            json.dump(combined_results, f)
        logger.info("Combined results saved successfully")
        return True
    except Exception as e:
        logger.error(f"Error saving combined results: {str(e)}")
        return False


def update_combined_results(model_name, track_name, track_data):
    """Update the combined results file with new track data"""
    try:
        # Load existing combined results
        combined_results = load_combined_results()

        # Initialize model entry if it doesn't exist
        if model_name not in combined_results:
            combined_results[model_name] = {}

        # Add or update track data
        combined_results[model_name][track_name] = track_data

        # Write updated results back to file
        save_combined_results(combined_results)
        return True
    except Exception as e:
        logger.error(f"Error updating combined results: {str(e)}")
        return False


def check_track_evaluated(model_name, track_name):
    """Check if a track has already been evaluated for a specific model"""
    combined_results = load_combined_results()
    return model_name in combined_results and track_name in combined_results[model_name]


def get_track_results(model_name, track_name):
    """Get the evaluation results for a specific track and model"""
    combined_results = load_combined_results()
    if model_name in combined_results and track_name in combined_results[model_name]:
        return combined_results[model_name][track_name]
    return None


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

        # Create a temporary directory for separation files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Perform separation if needed
            logger.info("Performing separation...")
            separator = Separator(output_dir=temp_dir)
            separator.load_model(model_filename=test_model)
            separator.separate(os.path.join(track_path, "mixture.wav"), custom_output_names=stem_mapping)

            # Check which stems were actually created and pair them appropriately
            available_stems = {}
            stem_pairs = {"drums": "nodrums", "bass": "nobass", "other": "noother", "vocals": "instrumental"}

            for main_stem, no_stem in stem_pairs.items():
                # Construct full file paths for both the isolated stem and its complement
                main_path = os.path.join(temp_dir, f"{main_stem}.wav")
                no_stem_path = os.path.join(temp_dir, f"{no_stem}.wav")

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
            # Use the temp directory for intermediate results
            scores = museval.eval_mus_track(track, estimates, output_dir=temp_dir, mode="v4")

            # Move only the final results file to the permanent location
            os.makedirs(output_dir, exist_ok=True)
            test_results = os.path.join(temp_dir, "test", f"{track_name}.json")
            train_results = os.path.join(temp_dir, "train", f"{track_name}.json")

            if os.path.exists(test_results):
                with open(test_results, "r") as f:
                    results_data = json.load(f)
                with open(results_file, "w") as f:
                    json.dump(results_data, f)
            elif os.path.exists(train_results):
                with open(train_results, "r") as f:
                    results_data = json.load(f)
                with open(results_file, "w") as f:
                    json.dump(results_data, f)
            # No need to remove directories as the temp directory will be automatically cleaned up

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


def check_inode_usage(path):
    """Check inode usage on the filesystem containing path"""
    import subprocess
    import re
    import sys

    result = subprocess.run(["df", "-i", path], capture_output=True, text=True)
    output = result.stdout
    logger.info(f"Current inode usage:\n{output}")

    # Parse the output to get inode usage percentage
    lines = output.strip().split("\n")
    if len(lines) >= 2:
        # The second line contains the actual data
        parts = lines[1].split()
        if len(parts) >= 8:  # macOS df -i format has 8 columns
            try:
                # On macOS, inode usage is in the 8th column as a percentage
                inode_usage_str = parts[7].rstrip("%")
                inode_usage_pct = int(inode_usage_str)

                # Also extract the actual inode numbers for better reporting
                iused = int(parts[5])
                ifree = int(parts[6])
                total_inodes = iused + ifree

                logger.info(f"Inode usage: {iused:,}/{total_inodes:,} ({inode_usage_pct}%)")

                if inode_usage_pct >= 100:
                    logger.critical("CRITICAL: Inode usage is at 100%! Cannot continue processing.")
                    logger.critical("Please free up inodes before continuing.")
                    sys.exit(1)
                elif inode_usage_pct > 90:
                    logger.warning(f"WARNING: High inode usage ({inode_usage_pct}%)!")

                return inode_usage_pct
            except (ValueError, IndexError) as e:
                logger.error(f"Error parsing inode usage: {str(e)}")

    # If we couldn't parse the output, try a different approach
    try:
        # Try using a more direct command to get inode information
        fs_stat = subprocess.run(["stat", "-f", "-c", "%d %i %f", path], capture_output=True, text=True)
        if fs_stat.returncode == 0:
            # Format: total_inodes used_inodes free_inodes
            stats = fs_stat.stdout.strip().split()
            if len(stats) >= 3:
                total = int(stats[0])
                used = int(stats[1])
                free = int(stats[2])
                usage_pct = (used / total) * 100 if total > 0 else 0

                logger.info(f"Inode usage (alternate method): {used:,}/{total:,} ({usage_pct:.1f}%)")

                if usage_pct >= 100:
                    logger.critical("CRITICAL: Inode usage is at 100%! Cannot continue processing.")
                    logger.critical("Please free up inodes before continuing.")
                    sys.exit(1)
                elif usage_pct > 90:
                    logger.warning(f"WARNING: High inode usage ({usage_pct:.1f}%)!")

                return usage_pct
    except Exception as e:
        logger.error(f"Error getting filesystem stats: {str(e)}")

    return None


def main():
    # Add command line argument parsing for dry run mode
    parser = argparse.ArgumentParser(description="Run model evaluation on MUSDB18 dataset")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode (no writes)")
    args = parser.parse_args()

    if args.dry_run:
        logger.info("*** RUNNING IN DRY-RUN MODE - NO DATA WILL BE MODIFIED ***")

    logger.info("Starting model evaluation script...")
    os.makedirs(RESULTS_PATH, exist_ok=True)

    # Check inode usage at start
    check_inode_usage(RESULTS_PATH)

    # Load existing results if available
    combined_results = {}
    if os.path.exists(COMBINED_RESULTS_PATH):
        logger.info("Loading existing combined results...")
        with open(COMBINED_RESULTS_PATH) as f:
            combined_results = json.load(f)

    # Load existing museval results
    museval_results = load_combined_results()
    logger.info(f"Loaded combined museval results with {len(museval_results)} models")

    # In dry-run mode, print some stats about the loaded data
    if args.dry_run:
        for model_name, tracks in museval_results.items():
            logger.info(f"Model {model_name} has {len(tracks)} evaluated tracks")
            if len(tracks) > 0:
                sample_track = next(iter(tracks))
                logger.info(f"  Sample track: {sample_track}")

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
    if not args.dry_run:
        os.makedirs(os.path.dirname(COMBINED_RESULTS_PATH), exist_ok=True)
        with open(COMBINED_RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(combined_results, f, indent=2)
        logger.info("Model stem information saved")
    else:
        logger.info("[DRY RUN] Would have saved model stem information")

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

                # Check if track already evaluated using the combined results file
                if check_track_evaluated(test_model, track_name):
                    logger.info(f"Skipping already evaluated track {track_name} for model: {test_model}")
                    continue

                # Process the model
                logger.info(f"Processing model: {test_model}")

                if args.dry_run:
                    logger.info(f"[DRY RUN] Would evaluate track {track_name} with model {test_model}")
                    continue

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

                # Save results after each model
                if not args.dry_run:
                    os.makedirs(os.path.dirname(COMBINED_RESULTS_PATH), exist_ok=True)
                    with open(COMBINED_RESULTS_PATH, "w", encoding="utf-8") as f:
                        json.dump(combined_results, f, indent=2)
                    logger.info(f"Updated combined results file with {test_model} - {track_name}")
                else:
                    logger.info(f"[DRY RUN] Would have updated combined results for {test_model} - {track_name}")

                # Check inode usage periodically
                if (len(combined_results[test_model]["track_scores"]) % 10) == 0:
                    check_inode_usage(RESULTS_PATH)

    logger.info("Evaluation complete")
    # Final inode usage check
    check_inode_usage(RESULTS_PATH)
    return 0


if __name__ == "__main__":
    exit(main())
