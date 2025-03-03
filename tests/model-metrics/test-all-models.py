#!/usr/bin/env python
import os
import time
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


def check_disk_usage(path):
    """Check inode usage and disk space on the filesystem containing path"""
    import subprocess
    import sys

    # Check disk space first
    result = subprocess.run(["df", "-h", path], capture_output=True, text=True)
    output = result.stdout
    logger.info(f"Current disk usage:\n{output}")

    # Parse the output to get disk usage percentage
    lines = output.strip().split("\n")
    if len(lines) >= 2:
        parts = lines[1].split()
        if len(parts) >= 5:
            try:
                # Extract disk usage percentage
                disk_usage_str = parts[4].rstrip("%")
                disk_usage_pct = int(disk_usage_str)

                logger.info(f"Disk usage: {disk_usage_pct}%")

                if disk_usage_pct >= 99:
                    logger.critical("CRITICAL: Disk is almost full (>99%)! Cannot continue processing.")
                    logger.critical("Please free up disk space before continuing.")
                    sys.exit(1)
                elif disk_usage_pct > 95:
                    logger.warning(f"WARNING: High disk usage ({disk_usage_pct}%)!")
            except (ValueError, IndexError) as e:
                logger.error(f"Error parsing disk usage: {str(e)}")

    # Now check inode usage
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

                # Skip inode check for exFAT or similar filesystems
                if total_inodes <= 1:
                    logger.info("Filesystem appears to be exFAT or similar (no real inode tracking). Skipping inode check.")
                    return None

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

    return None


def get_evaluated_track_count(model_name, museval_results):
    """Get the number of tracks evaluated for a specific model"""
    if model_name in museval_results:
        return len(museval_results[model_name])
    return 0


def get_most_evaluated_tracks(museval_results, min_count=10):
    """Get tracks that have been evaluated for the most models"""
    track_counts = {}

    # Count how many models have evaluated each track
    for model_name, tracks in museval_results.items():
        for track_name in tracks:
            if track_name not in track_counts:
                track_counts[track_name] = 0
            track_counts[track_name] += 1

    # Sort tracks by evaluation count (descending)
    sorted_tracks = sorted(track_counts.items(), key=lambda x: x[1], reverse=True)

    # Return tracks that have been evaluated at least min_count times
    return [track for track, count in sorted_tracks if count >= min_count]


def main():
    # Add command line argument parsing for dry run mode
    parser = argparse.ArgumentParser(description="Run model evaluation on MUSDB18 dataset")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode (no writes)")
    parser.add_argument("--max-tracks", type=int, default=10, help="Maximum number of tracks to evaluate per model")
    parser.add_argument("--max-models", type=int, default=None, help="Maximum number of models to evaluate")
    args = parser.parse_args()

    # Track start time for progress reporting
    start_time = time.time()

    # Create a results cache manager
    class ResultsCache:
        def __init__(self):
            self.results = load_combined_results()
            self.last_update_time = time.time()

        def get_results(self, force=False):
            current_time = time.time()
            # Only reload from disk every 5 minutes unless forced
            if force or (current_time - self.last_update_time) > 300:
                self.results = load_combined_results()
                self.last_update_time = current_time
            return self.results

    results_cache = ResultsCache()

    # Helper function for logging with elapsed time
    def log_with_time(message, level=logging.INFO):
        elapsed = time.time() - start_time
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
        logger.log(level, f"[{time_str}] {message}")

    if args.dry_run:
        log_with_time("*** RUNNING IN DRY-RUN MODE - NO DATA WILL BE MODIFIED ***")

    log_with_time("Starting model evaluation script...")
    os.makedirs(RESULTS_PATH, exist_ok=True)

    # Check disk space and inode usage at start
    check_disk_usage(RESULTS_PATH)

    # Load existing results if available
    combined_results = {}
    if os.path.exists(COMBINED_RESULTS_PATH):
        log_with_time("Loading existing combined results...")
        with open(COMBINED_RESULTS_PATH) as f:
            combined_results = json.load(f)

    # Get initial museval results
    museval_results = results_cache.get_results()
    log_with_time(f"Loaded combined museval results with {len(museval_results)} models")

    # Get the most commonly evaluated tracks
    common_tracks = get_most_evaluated_tracks(museval_results)
    log_with_time(f"Found {len(common_tracks)} commonly evaluated tracks")

    # Initialize MUSDB
    log_with_time("Initializing MUSDB database...")
    mus = musdb.DB(root=MUSDB_PATH, is_wav=True)

    # Create a prioritized list of tracks
    all_tracks = []
    for track in mus.tracks:
        # Check if this is a commonly evaluated track
        is_common = track.name in common_tracks
        all_tracks.append({"name": track.name, "path": os.path.dirname(track.path), "is_common": is_common})

    # Sort tracks by whether they're commonly evaluated
    all_tracks.sort(key=lambda t: 0 if t["is_common"] else 1)

    # Get list of all available models
    log_with_time("Getting list of available models...")
    separator = Separator()
    models_by_type = separator.list_supported_model_files()

    # Flatten the models list and prioritize them
    all_models = []
    for model_type, models in models_by_type.items():
        for model_name, model_info in models.items():
            filename = model_info.get("filename")
            if filename:
                # Count how many tracks have been evaluated for this model
                evaluated_count = get_evaluated_track_count(filename, museval_results)

                # Determine if this is a roformer model
                is_roformer = "roformer" in model_name.lower()

                # Add to the list with priority information
                all_models.append({"name": model_name, "filename": filename, "type": model_type, "info": model_info, "evaluated_count": evaluated_count, "is_roformer": is_roformer})

    # Sort models by priority:
    # 1. Roformer models with fewer than max_tracks evaluations
    # 2. Other models with fewer than max_tracks evaluations
    # 3. Roformer models with more evaluations
    # 4. Other models with more evaluations
    all_models.sort(
        key=lambda m: (
            0 if m["is_roformer"] and m["evaluated_count"] < args.max_tracks else 1 if not m["is_roformer"] and m["evaluated_count"] < args.max_tracks else 2 if m["is_roformer"] else 3,
            m["evaluated_count"],  # Secondary sort by number of evaluations (ascending)
        )
    )

    # Log the prioritized models
    log_with_time(f"Prioritized {len(all_models)} models for evaluation:")
    for i, model in enumerate(all_models[:10]):  # Show top 10
        log_with_time(f"{i+1}. {model['name']} ({model['filename']}) - {model['evaluated_count']} tracks evaluated, roformer: {model['is_roformer']}")

    if len(all_models) > 10:
        log_with_time(f"... and {len(all_models) - 10} more models")

    # Limit the number of models if specified
    if args.max_models:
        all_models = all_models[: args.max_models]
        log_with_time(f"Limited to {args.max_models} models for this run")

    # Process models according to priority
    model_idx = 0
    while model_idx < len(all_models):
        model = all_models[model_idx]
        model_name = model["name"]
        model_filename = model["filename"]
        model_type = model["type"]

        progress_pct = (model_idx + 1) / len(all_models) * 100
        log_with_time(f"\n=== Processing model {model_idx+1}/{len(all_models)} ({progress_pct:.1f}%): {model_name} ({model_filename}) ===")

        # Initialize model entry if it doesn't exist
        if model_filename not in combined_results:
            log_with_time(f"Initializing new entry for {model_filename}")
            combined_results[model_filename] = {"model_name": model_name, "track_scores": [], "median_scores": {}, "stems": [], "target_stem": None}

        # Try to load the model to get stem information
        try:
            separator.load_model(model_filename=model_filename)
            model_data = separator.model_instance.model_data

            # Extract stem information (similar to your existing code)
            # ... (keep your existing stem extraction logic here)

        except Exception as e:
            log_with_time(f"Error loading model {model_filename}: {str(e)}", logging.ERROR)
            logger.exception("Full exception details:")
            model_idx += 1
            continue

        # Count how many tracks have been evaluated for this model
        # Use the cached results
        evaluated_count = get_evaluated_track_count(model_filename, results_cache.get_results())

        # Determine how many more tracks to evaluate
        tracks_to_evaluate = max(0, args.max_tracks - evaluated_count)

        if tracks_to_evaluate == 0:
            log_with_time(f"Model {model_name} already has {evaluated_count} tracks evaluated (>= {args.max_tracks}). Skipping.")
            model_idx += 1
            continue

        log_with_time(f"Will evaluate up to {tracks_to_evaluate} tracks for model {model_name}")

        # Process tracks for this model
        tracks_processed = 0
        for track in all_tracks:
            # Skip if we've processed enough tracks for this model
            if tracks_processed >= tracks_to_evaluate:
                break

            track_name = track["name"]
            track_path = track["path"]

            # Skip if track already evaluated for this model
            # Use the cached results
            if model_filename in results_cache.get_results() and track_name in results_cache.get_results()[model_filename]:
                log_with_time(f"Skipping already evaluated track {track_name} for model: {model_filename}")
                continue

            log_with_time(f"Processing track: {track_name} for model: {model_filename}")

            if args.dry_run:
                log_with_time(f"[DRY RUN] Would evaluate track {track_name} with model {model_filename}")
                tracks_processed += 1
                continue

            try:
                _, model_results = evaluate_track(track_name, track_path, model_filename, mus)
                if model_results:
                    combined_results[model_filename]["track_scores"].append(model_results)
                    tracks_processed += 1
                else:
                    log_with_time(f"Skipping model {model_filename} for track {track_name} due to no evaluatable stems")
            except Exception as e:
                log_with_time(f"Error evaluating model {model_filename} with track {track_name}: {str(e)}", logging.ERROR)
                logger.exception(f"Exception details: ", exc_info=e)
                continue

            # Update and save results
            if combined_results[model_filename]["track_scores"]:
                median_scores = calculate_median_scores(combined_results[model_filename]["track_scores"])
                combined_results[model_filename]["median_scores"] = median_scores

            # Save results after each track
            if not args.dry_run:
                os.makedirs(os.path.dirname(COMBINED_RESULTS_PATH), exist_ok=True)
                with open(COMBINED_RESULTS_PATH, "w", encoding="utf-8") as f:
                    json.dump(combined_results, f, indent=2)
                log_with_time(f"Updated combined results file with {model_filename} - {track_name}")

                # Force update the cache after saving
                results_cache.get_results(force=True)
            else:
                log_with_time(f"[DRY RUN] Would have updated combined results for {model_filename} - {track_name}")

            # Check disk space periodically
            check_disk_usage(RESULTS_PATH)

        log_with_time(f"Completed processing {tracks_processed} tracks for model {model_name}")

        # If we're processing a non-roformer model, check if there are roformer models that need evaluation
        if not model["is_roformer"]:
            # Find roformer models that still need more evaluations
            # Use the cached results
            roformer_models_needing_eval = []
            for i, m in enumerate(all_models[model_idx + 1 :], start=model_idx + 1):
                if m["is_roformer"]:
                    eval_count = get_evaluated_track_count(m["filename"], results_cache.get_results())
                    if eval_count < args.max_tracks:
                        roformer_models_needing_eval.append((i, m))

            if roformer_models_needing_eval:
                log_with_time(f"Found {len(roformer_models_needing_eval)} roformer models that still need evaluation. Reprioritizing...")

                # Move these models to the front of the remaining queue
                for offset, (i, m) in enumerate(roformer_models_needing_eval):
                    # Adjust index for models we've already moved
                    adjusted_idx = i - offset
                    # Move this model right after the current one
                    all_models.insert(model_idx + 1, all_models.pop(adjusted_idx))

                log_with_time("Reprioritization complete. Continuing with highest priority model.")

        # Move to the next model
        model_idx += 1

    log_with_time("Evaluation complete")
    # Final disk space check
    check_disk_usage(RESULTS_PATH)
    return 0


if __name__ == "__main__":
    exit(main())
