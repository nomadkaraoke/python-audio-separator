#!/usr/bin/env python
import os
import time
import museval
import numpy as np
import soundfile as sf
from audio_separator.separator import Separator
import json
from json import JSONEncoder
import logging
import musdb
from decimal import Decimal
import tempfile
import argparse


# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Custom JSON Encoder to handle Decimal types
class DecimalEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)


MUSDB_PATH = "/Volumes/Nomad4TBOne/python-audio-separator/tests/model-metrics/datasets/musdb18hq"
RESULTS_PATH = "/Volumes/Nomad4TBOne/python-audio-separator/tests/model-metrics/results"
COMBINED_RESULTS_PATH = "/Users/andrew/Projects/python-audio-separator/audio_separator/models-scores.json"
COMBINED_MUSEVAL_RESULTS_PATH = "/Volumes/Nomad4TBOne/python-audio-separator/tests/model-metrics/results/combined-museval-results.json"
STOP_SIGNAL_PATH = "/Volumes/Nomad4TBOne/python-audio-separator/tests/model-metrics/stop-signal"


def load_combined_results():
    """Load the combined museval results file"""
    if os.path.exists(COMBINED_MUSEVAL_RESULTS_PATH):
        logger.info("Loading combined museval results...")
        try:
            with open(COMBINED_MUSEVAL_RESULTS_PATH, "r") as f:
                # Use a custom parser to handle Decimal values
                def decimal_parser(dct):
                    for k, v in dct.items():
                        if isinstance(v, str) and v.replace(".", "").isdigit():
                            try:
                                dct[k] = float(v)
                            except (ValueError, TypeError):
                                pass
                    return dct

                return json.load(f, object_hook=decimal_parser)
        except Exception as e:
            logger.error(f"Error loading combined results: {str(e)}")
            # Try to load a backup file if it exists
            backup_path = COMBINED_MUSEVAL_RESULTS_PATH + ".backup"
            if os.path.exists(backup_path):
                logger.info("Attempting to load backup file...")
                try:
                    with open(backup_path, "r") as f:
                        return json.load(f, object_hook=decimal_parser)
                except Exception as backup_e:
                    logger.error(f"Error loading backup file: {str(backup_e)}")
            return {}
    else:
        logger.info("No combined results file found, creating new one")
        return {}


def save_combined_results(combined_results):
    """Save the combined museval results file"""
    logger.info("Saving combined museval results...")
    try:
        # Create a backup of the existing file if it exists
        if os.path.exists(COMBINED_MUSEVAL_RESULTS_PATH):
            backup_path = COMBINED_MUSEVAL_RESULTS_PATH + ".backup"
            try:
                with open(COMBINED_MUSEVAL_RESULTS_PATH, "r") as src, open(backup_path, "w") as dst:
                    dst.write(src.read())
            except Exception as e:
                logger.error(f"Error creating backup file: {str(e)}")

        # Save the new results using the custom encoder
        with open(COMBINED_MUSEVAL_RESULTS_PATH, "w") as f:
            json.dump(combined_results, f, cls=DecimalEncoder, indent=2)
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


def get_track_duration(track_path):
    """Get the duration of a track in minutes"""
    try:
        mixture_path = os.path.join(track_path, "mixture.wav")
        info = sf.info(mixture_path)
        return info.duration / 60.0  # Convert seconds to minutes
    except Exception as e:
        logger.error(f"Error getting track duration: {str(e)}")
        return 0.0


def evaluate_track(track_name, track_path, test_model, mus_db):
    """Evaluate a single track using a specific model"""
    logger.info(f"Evaluating track: {track_name} with model: {test_model}")

    # Get track duration in minutes
    track_duration_minutes = get_track_duration(track_path)
    logger.info(f"Track duration: {track_duration_minutes:.2f} minutes")

    # Initialize variables to track processing time
    processing_time = 0
    seconds_per_minute = 0

    # Create a basic result structure that will be returned even if evaluation fails
    basic_model_results = {"track_name": track_name, "scores": {}}

    # Check if evaluation results already exist in combined file
    museval_results = load_combined_results()
    if test_model in museval_results and track_name in museval_results[test_model]:
        logger.info("Found existing evaluation results in combined file...")
        track_data = museval_results[test_model][track_name]
        scores = museval.TrackStore(track_name)
        scores.scores = track_data

        # Try to extract existing speed metrics if available
        try:
            if isinstance(track_data, dict) and "targets" in track_data:
                for target in track_data["targets"]:
                    if "metrics" in target and "seconds_per_minute_m3" in target["metrics"]:
                        basic_model_results["scores"]["seconds_per_minute_m3"] = target["metrics"]["seconds_per_minute_m3"]
                        break
        except Exception:
            pass  # Ignore errors in extracting existing speed metrics
    else:
        # Expanded stem mapping to include "no-stem" outputs and custom stem formats
        stem_mapping = {
            # Standard stems
            "Vocals": "vocals",
            "Instrumental": "instrumental",
            "Drums": "drums",
            "Bass": "bass",
            "Other": "other",
            # No-stem variants
            "No Drums": "nodrums",
            "No Bass": "nobass",
            "No Other": "noother",
            # Custom stem formats (with hyphens)
            "Drum-Bass": "drumbass",
            "No Drum-Bass": "nodrumbass",
            "Vocals-Other": "vocalsother",
            "No Vocals-Other": "novocalsother",
        }

        # Create a temporary directory for separation files
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"Using temporary directory: {temp_dir}")

            # Measure separation time
            start_time = time.time()

            # Perform separation
            logger.info("Performing separation...")
            separator = Separator(output_dir=temp_dir)
            separator.load_model(model_filename=test_model)
            separator.separate(os.path.join(track_path, "mixture.wav"), custom_output_names=stem_mapping)

            # Calculate processing time
            processing_time = time.time() - start_time
            seconds_per_minute = processing_time / track_duration_minutes if track_duration_minutes > 0 else 0
            logger.info(f"Separation completed in {processing_time:.2f} seconds")
            logger.info(f"Processing speed: {seconds_per_minute:.2f} seconds per minute of audio")

            # Always add the speed metric to our basic results
            basic_model_results["scores"]["seconds_per_minute_m3"] = round(seconds_per_minute, 1)

            # Check which stems were actually created
            wav_files = [f for f in os.listdir(temp_dir) if f.endswith(".wav")]
            logger.info(f"Found WAV files: {wav_files}")

            # Determine if this is a standard vocal/instrumental model that can be evaluated with museval
            standard_model = False
            if len(wav_files) == 2:
                # Check if one of the files is named vocals.wav or instrumental.wav
                if "vocals.wav" in wav_files and "instrumental.wav" in wav_files:
                    standard_model = True
                    logger.info("Detected standard vocals/instrumental model, will run museval evaluation")

            # If not a standard model, skip museval evaluation and just return speed metrics
            if not standard_model:
                logger.info(f"Non-standard stem configuration detected for model {test_model}, skipping museval evaluation")

                # Store the speed metric in the combined results
                if test_model not in museval_results:
                    museval_results[test_model] = {}

                # Create a minimal structure for the speed metric
                minimal_results = {"targets": [{"name": "speed_metrics_only", "metrics": {"seconds_per_minute_m3": round(seconds_per_minute, 1)}}]}

                museval_results[test_model][track_name] = minimal_results
                save_combined_results(museval_results)

                return None, basic_model_results

            # For standard models, proceed with museval evaluation
            available_stems = {}
            available_stems["vocals"] = os.path.join(temp_dir, "vocals.wav")
            available_stems["accompaniment"] = os.path.join(temp_dir, "instrumental.wav")

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
            try:
                scores = museval.eval_mus_track(track, estimates, output_dir=temp_dir, mode="v4")

                # Add the speed metric to the scores
                if not hasattr(scores, "speed_metric_added"):
                    for target in scores.scores["targets"]:
                        if "metrics" not in target:
                            target["metrics"] = {}
                        target["metrics"]["seconds_per_minute_m3"] = round(seconds_per_minute, 1)
                    scores.speed_metric_added = True

                # Update the combined results file with the new evaluation
                if test_model not in museval_results:
                    museval_results[test_model] = {}
                museval_results[test_model][track_name] = scores.scores
                save_combined_results(museval_results)
            except Exception as e:
                logger.error(f"Error during museval evaluation: {str(e)}")
                logger.exception("Evaluation exception details:")
                # Return basic results with just the speed metric
                return None, basic_model_results

    try:
        # Only process museval results if we have them
        if "scores" in locals() and scores is not None:
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

            # Add the seconds_per_minute_m3 metric if it was calculated
            if processing_time > 0 and track_duration_minutes > 0:
                model_results["scores"]["seconds_per_minute_m3"] = round(seconds_per_minute, 1)

            return scores, model_results if model_results["scores"] else basic_model_results
        else:
            # If we don't have scores, just return the basic results with speed metrics
            return None, basic_model_results

    except Exception as e:
        logger.error(f"Error processing evaluation results: {str(e)}")
        logger.exception("Results processing exception details:")
        # Return basic results with just the speed metric
        return None, basic_model_results


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
        "seconds_per_minute_m3": [],
    }

    # Collect all scores for each stem and metric
    for track_score in track_scores:
        if track_score is not None and "scores" in track_score:
            # Process audio quality metrics
            for stem, metrics in track_score["scores"].items():
                if stem in stem_metrics and stem != "seconds_per_minute_m3":
                    for metric, value in metrics.items():
                        stem_metrics[stem][metric].append(value)

            # Process speed metric separately
            if "seconds_per_minute_m3" in track_score["scores"]:
                stem_metrics["seconds_per_minute_m3"].append(track_score["scores"]["seconds_per_minute_m3"])

    # Calculate medians for each stem and metric
    median_scores = {}
    for stem, metrics in stem_metrics.items():
        if stem != "seconds_per_minute_m3" and any(metrics.values()):  # Only include stems that have scores
            median_scores[stem] = {metric: float(f"{np.median(values):.6g}") for metric, values in metrics.items() if values}  # Only include metrics that have values

    # Add median speed metric if available
    if stem_metrics["seconds_per_minute_m3"]:
        median_scores["seconds_per_minute_m3"] = round(np.median(stem_metrics["seconds_per_minute_m3"]), 1)

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


def generate_summary_statistics(
    start_time, models_processed, tracks_processed, models_with_new_data, tracks_evaluated, total_processing_time, fastest_model=None, slowest_model=None, combined_results_path=None, is_dry_run=False
):
    """Generate a summary of the script's execution"""
    end_time = time.time()
    total_runtime = end_time - start_time

    # Format the runtime
    hours, remainder = divmod(total_runtime, 3600)
    minutes, seconds = divmod(remainder, 60)
    runtime_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

    # Build the summary
    summary = [
        "=" * 80,
        "DRY RUN SUMMARY - PREVIEW ONLY" if is_dry_run else "EXECUTION SUMMARY",
        "=" * 80,
        f"Total runtime: {runtime_str}",
        f"Models {'that would be' if is_dry_run else ''} processed: {models_processed}",
        f"Models {'that would receive' if is_dry_run else 'with'} new data: {len(models_with_new_data)}",
        f"Total tracks {'that would be' if is_dry_run else ''} evaluated: {tracks_evaluated}",
        f"Average tracks per model: {tracks_evaluated / len(models_with_new_data) if models_with_new_data else 0:.2f}",
    ]

    if fastest_model:
        summary.append(f"Fastest model: {fastest_model['name']} ({fastest_model['speed']:.2f} seconds per minute)")

    if slowest_model:
        summary.append(f"Slowest model: {slowest_model['name']} ({slowest_model['speed']:.2f} seconds per minute)")

    if total_processing_time > 0:
        summary.append(f"Total audio processing time: {total_processing_time:.2f} seconds")

    if combined_results_path and os.path.exists(combined_results_path):
        file_size = os.path.getsize(combined_results_path) / (1024 * 1024)  # Size in MB
        summary.append(f"Results file size: {file_size:.2f} MB")

    # Add models with new data
    if models_with_new_data:
        summary.append(f"\nModels {'that would receive' if is_dry_run else 'with'} new evaluation data:")
        for model_name in models_with_new_data:
            summary.append(f"- {model_name}")

    # Add dry run disclaimer if needed
    if is_dry_run:
        summary.append("\nNOTE: This is a dry run summary. No actual changes were made.")
        summary.append("Run without --dry-run to perform actual evaluations.")

    summary.append("=" * 80)
    return "\n".join(summary)


def check_stop_signal():
    """Check if the stop signal file exists"""
    if os.path.exists(STOP_SIGNAL_PATH):
        logger.info("Stop signal detected at: " + STOP_SIGNAL_PATH)
        return True
    return False


def main():
    # Add command line argument parsing for dry run mode
    parser = argparse.ArgumentParser(description="Run model evaluation on MUSDB18 dataset")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode (no writes)")
    parser.add_argument("--max-tracks", type=int, default=10, help="Maximum number of tracks to evaluate per model")
    parser.add_argument("--max-models", type=int, default=None, help="Maximum number of models to evaluate")
    args = parser.parse_args()

    # Remove any existing stop signal file at start
    if os.path.exists(STOP_SIGNAL_PATH):
        os.remove(STOP_SIGNAL_PATH)
        logger.info("Removed existing stop signal file")

    # Track start time for progress reporting
    start_time = time.time()

    # Statistics tracking
    models_processed = 0
    tracks_processed = 0
    models_with_new_data = set()
    total_processing_time = 0
    fastest_model = {"name": "", "speed": float("inf")}  # Initialize with infinity for comparison
    slowest_model = {"name": "", "speed": 0}  # Initialize with zero for comparison

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
    stop_requested = False
    while model_idx < len(all_models):
        # Check for stop signal before processing each model
        if check_stop_signal():
            log_with_time("Stop signal detected. Will finish current model's tracks and then exit.")
            stop_requested = True

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
            # Check for stop signal before each track if we haven't already detected it
            if not stop_requested and check_stop_signal():
                log_with_time("Stop signal detected. Will finish current track and then exit.")
                stop_requested = True

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
                models_with_new_data.add(model_filename)

                # Estimate processing time based on model type for dry run
                # This is a rough estimate - roformer models are typically slower
                estimated_speed = 30.0  # Default estimate: 30 seconds per minute
                if "roformer" in model_name.lower():
                    estimated_speed = 45.0  # Roformer models are typically slower
                elif "umx" in model_name.lower():
                    estimated_speed = 20.0  # UMX models are typically faster

                # Update statistics with estimated values
                total_processing_time += estimated_speed

                # Track fastest and slowest models based on estimates
                if estimated_speed < fastest_model["speed"]:
                    fastest_model = {"name": model_name, "speed": estimated_speed}
                if estimated_speed > slowest_model["speed"]:
                    slowest_model = {"name": model_name, "speed": estimated_speed}

                continue

            try:
                result = evaluate_track(track_name, track_path, model_filename, mus)

                # Unpack the result safely
                if result and isinstance(result, tuple) and len(result) == 2:
                    _, model_results = result
                else:
                    model_results = None

                # Process the results if they exist and are valid
                if model_results is not None and isinstance(model_results, dict):
                    combined_results[model_filename]["track_scores"].append(model_results)
                    tracks_processed += 1
                    models_with_new_data.add(model_filename)

                    # Track processing time statistics - safely access nested dictionaries
                    scores = model_results.get("scores", {})
                    if isinstance(scores, dict):
                        speed = scores.get("seconds_per_minute_m3")
                        if speed is not None:
                            total_processing_time += speed  # Accumulate total processing time

                            # Track fastest and slowest models
                            if speed < fastest_model["speed"]:
                                fastest_model = {"name": model_name, "speed": speed}
                            if speed > slowest_model["speed"]:
                                slowest_model = {"name": model_name, "speed": speed}
                else:
                    log_with_time(f"Skipping model {model_filename} for track {track_name} due to no evaluatable stems or invalid results")
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

        # If stop was requested, exit after completing the current model
        if stop_requested:
            log_with_time("Stop signal processed. Generating final summary before exit.")
            break

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
        models_processed += 1

    log_with_time("Evaluation complete")
    # Final disk space check
    check_disk_usage(RESULTS_PATH)

    # Generate and display summary statistics
    # Reset fastest/slowest models if they weren't updated
    if fastest_model["speed"] == float("inf"):
        fastest_model = None
    if slowest_model["speed"] == 0:
        slowest_model = None

    summary = generate_summary_statistics(
        start_time=start_time,
        models_processed=models_processed,
        tracks_processed=tracks_processed,
        models_with_new_data=models_with_new_data,
        tracks_evaluated=tracks_processed,
        total_processing_time=total_processing_time,
        fastest_model=fastest_model,
        slowest_model=slowest_model,
        combined_results_path=COMBINED_RESULTS_PATH,
        is_dry_run=args.dry_run,
    )

    log_with_time("\n" + summary)

    # Also write summary to a log file
    summary_filename = "dry_run_summary.log" if args.dry_run else "evaluation_summary.log"
    if stop_requested:
        summary_filename = "stopped_" + summary_filename
    summary_log_path = os.path.join(os.path.dirname(COMBINED_RESULTS_PATH), summary_filename)
    with open(summary_log_path, "w") as f:
        f.write(f"{'Dry run' if args.dry_run else 'Evaluation'} {'(stopped early)' if stop_requested else ''} completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(summary)

    log_with_time(f"Summary written to {summary_log_path}")

    # Clean up stop signal file if it exists
    if os.path.exists(STOP_SIGNAL_PATH):
        os.remove(STOP_SIGNAL_PATH)
        log_with_time("Removed stop signal file")

    return 0 if not stop_requested else 2  # Return different exit code if stopped early


if __name__ == "__main__":
    exit(main())
