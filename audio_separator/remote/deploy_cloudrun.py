"""
Audio Separator API - Cloud Run GPU Deployment

A FastAPI service for separating vocals from instrumental tracks using audio-separator,
deployed on Google Cloud Run with L4 GPU acceleration.

This is the GCP equivalent of deploy_modal.py — same API contract, different infrastructure.
Models are downloaded from GCS on startup and cached in the container's local filesystem.

Usage with Remote CLI:
1. Install audio-separator package: pip install audio-separator
2. Set environment variable: export AUDIO_SEPARATOR_API_URL="https://your-cloudrun-url.run.app"
3. Use the remote CLI:
   - audio-separator-remote separate song.mp3
   - audio-separator-remote separate song.mp3 --model UVR-MDX-NET-Inst_HQ_4
   - audio-separator-remote status <task_id>
   - audio-separator-remote models
   - audio-separator-remote download <task_id> <filename>
"""

import asyncio
import hashlib
import json
import logging
import os
import shutil
import threading
import traceback
import typing
import uuid
from importlib.metadata import version
from typing import Optional
from urllib.parse import quote

import filetype
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import PlainTextResponse
from starlette.responses import Response as StarletteResponse

logger = logging.getLogger("audio-separator-api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# Constants
MODEL_DIR = os.environ.get("MODEL_DIR", "/models")
STORAGE_DIR = os.environ.get("STORAGE_DIR", "/tmp/storage")
MODEL_BUCKET = os.environ.get("MODEL_BUCKET", "")
PORT = int(os.environ.get("PORT", "8080"))

# In-memory job status tracking (one instance handles one job at a time on Cloud Run GPU)
job_status_store: dict[str, dict] = {}

# Track model readiness
models_ready = False


def generate_file_hash(filename: str) -> str:
    """Generate a short, stable hash for a filename to use in download URLs."""
    return hashlib.sha256(filename.encode("utf-8")).hexdigest()[:16]


try:
    AUDIO_SEPARATOR_VERSION = version("audio-separator")
except Exception:
    AUDIO_SEPARATOR_VERSION = "unknown"


def download_models_from_gcs():
    """Download models from GCS bucket on startup."""
    global models_ready

    if not MODEL_BUCKET:
        logger.info("MODEL_BUCKET not set, skipping GCS model download (models will be downloaded on demand)")
        models_ready = True
        return

    try:
        from google.cloud import storage

        client = storage.Client()
        bucket = client.bucket(MODEL_BUCKET)
        blobs = list(bucket.list_blobs())

        os.makedirs(MODEL_DIR, exist_ok=True)

        for blob in blobs:
            local_path = os.path.join(MODEL_DIR, blob.name)
            if os.path.exists(local_path):
                # Check size to skip already-downloaded models
                if os.path.getsize(local_path) == blob.size:
                    logger.info(f"Model already cached: {blob.name} ({blob.size / 1024 / 1024:.1f} MB)")
                    continue

            logger.info(f"Downloading model: {blob.name} ({blob.size / 1024 / 1024:.1f} MB)")
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            blob.download_to_filename(local_path)
            logger.info(f"Downloaded: {blob.name}")

        models_ready = True
        logger.info(f"All models ready in {MODEL_DIR}")

    except Exception as e:
        logger.error(f"Failed to download models from GCS: {e}")
        # Still mark as ready — models can be downloaded on demand by Separator
        models_ready = True


def separate_audio_sync(
    audio_data: bytes,
    filename: str,
    task_id: str,
    models: Optional[list] = None,
    preset: Optional[str] = None,
    output_format: str = "flac",
    output_bitrate: Optional[str] = None,
    normalization_threshold: float = 0.9,
    amplification_threshold: float = 0.0,
    output_single_stem: Optional[str] = None,
    invert_using_spec: bool = False,
    sample_rate: int = 44100,
    use_soundfile: bool = False,
    use_autocast: bool = False,
    custom_output_names: Optional[dict] = None,
    # MDX parameters
    mdx_segment_size: int = 256,
    mdx_overlap: float = 0.25,
    mdx_batch_size: int = 1,
    mdx_hop_length: int = 1024,
    mdx_enable_denoise: bool = False,
    # VR parameters
    vr_batch_size: int = 1,
    vr_window_size: int = 512,
    vr_aggression: int = 5,
    vr_enable_tta: bool = False,
    vr_high_end_process: bool = False,
    vr_enable_post_process: bool = False,
    vr_post_process_threshold: float = 0.2,
    # Demucs parameters
    demucs_segment_size: str = "Default",
    demucs_shifts: int = 2,
    demucs_overlap: float = 0.25,
    demucs_segments_enabled: bool = True,
    # MDXC parameters
    mdxc_segment_size: int = 256,
    mdxc_override_model_segment_size: bool = False,
    mdxc_overlap: int = 8,
    mdxc_batch_size: int = 1,
    mdxc_pitch_shift: int = 0,
) -> dict:
    """Separate audio into stems. Runs synchronously (Cloud Run GPU handles one job at a time)."""
    from audio_separator.separator import Separator

    all_output_files = {}
    models_used = []

    def update_status(status: str, progress: int = 0, error: str = None, files: dict = None):
        status_data = {
            "task_id": task_id,
            "status": status,
            "progress": progress,
            "original_filename": filename,
            "models_used": models_used,
            "total_models": len(models) if models else 1,
            "current_model_index": 0,
            "files": files or {},
        }
        if error:
            status_data["error"] = error
        job_status_store[task_id] = status_data

    try:
        os.makedirs(f"{STORAGE_DIR}/outputs/{task_id}", exist_ok=True)
        output_dir = f"{STORAGE_DIR}/outputs/{task_id}"

        update_status("processing", 5)

        # Write uploaded file
        input_file_path = os.path.join(output_dir, filename)
        with open(input_file_path, "wb") as f:
            f.write(audio_data)

        update_status("processing", 10)

        # Build separator kwargs
        separator_kwargs = {
            "log_level": logging.INFO,
            "model_file_dir": MODEL_DIR,
            "output_dir": output_dir,
            "output_format": output_format,
            "output_bitrate": output_bitrate,
            "normalization_threshold": normalization_threshold,
            "amplification_threshold": amplification_threshold,
            "output_single_stem": output_single_stem,
            "invert_using_spec": invert_using_spec,
            "sample_rate": sample_rate,
            "use_soundfile": use_soundfile,
            "use_autocast": use_autocast,
            "mdx_params": {
                "hop_length": mdx_hop_length,
                "segment_size": mdx_segment_size,
                "overlap": mdx_overlap,
                "batch_size": mdx_batch_size,
                "enable_denoise": mdx_enable_denoise,
            },
            "vr_params": {
                "batch_size": vr_batch_size,
                "window_size": vr_window_size,
                "aggression": vr_aggression,
                "enable_tta": vr_enable_tta,
                "enable_post_process": vr_enable_post_process,
                "post_process_threshold": vr_post_process_threshold,
                "high_end_process": vr_high_end_process,
            },
            "demucs_params": {
                "segment_size": demucs_segment_size,
                "shifts": demucs_shifts,
                "overlap": demucs_overlap,
                "segments_enabled": demucs_segments_enabled,
            },
            "mdxc_params": {
                "segment_size": mdxc_segment_size,
                "batch_size": mdxc_batch_size,
                "overlap": mdxc_overlap,
                "override_model_segment_size": mdxc_override_model_segment_size,
                "pitch_shift": mdxc_pitch_shift,
            },
        }

        if preset:
            # Use ensemble preset — Separator handles model resolution
            separator_kwargs["ensemble_preset"] = preset
            logger.info(f"Using ensemble preset: {preset}")

            separator = Separator(**separator_kwargs)
            separator.load_model()  # Preset models loaded automatically
            models_used.append(f"preset:{preset}")

            update_status("processing", 50)
            output_files = separator.separate(input_file_path, custom_output_names=custom_output_names)

            if not output_files:
                error_msg = f"Separation with preset {preset} produced no output files"
                update_status("error", 0, error=error_msg)
                return {"task_id": task_id, "status": "error", "error": error_msg, "models_used": models_used}

            for f in output_files:
                fname = os.path.basename(f)
                all_output_files[generate_file_hash(fname)] = fname

        else:
            # Traditional multi-model processing (no ensembling)
            if models is None or len(models) == 0:
                models_to_run = [None]
            else:
                models_to_run = models

            total_models = len(models_to_run)

            for model_index, model_name in enumerate(models_to_run):
                base_progress = 10 + (model_index * 80 // total_models)
                model_progress_range = 80 // total_models

                logger.info(f"Processing model {model_index + 1}/{total_models}: {model_name or 'default'}")
                update_status("processing", base_progress + (model_progress_range // 4))

                separator = Separator(**separator_kwargs)

                update_status("processing", base_progress + (model_progress_range // 2))
                if model_name:
                    separator.load_model(model_name)
                    models_used.append(model_name)
                else:
                    separator.load_model()
                    models_used.append("default")

                update_status("processing", base_progress + (3 * model_progress_range // 4))

                model_custom_output_names = None
                if total_models > 1 and custom_output_names:
                    model_suffix = f"_{models_used[-1].replace('.', '_').replace('/', '_')}"
                    model_custom_output_names = {stem: f"{name}{model_suffix}" for stem, name in custom_output_names.items()}
                elif custom_output_names:
                    model_custom_output_names = custom_output_names

                output_files = separator.separate(input_file_path, custom_output_names=model_custom_output_names)

                if not output_files:
                    error_msg = f"Separation with model {models_used[-1]} produced no output files"
                    update_status("error", 0, error=error_msg)
                    return {"task_id": task_id, "status": "error", "error": error_msg, "models_used": models_used}

                for f in output_files:
                    fname = os.path.basename(f)
                    all_output_files[generate_file_hash(fname)] = fname

        update_status("completed", 100, files=all_output_files)
        logger.info(f"Separation completed. {len(all_output_files)} output files.")
        return {"task_id": task_id, "status": "completed", "files": all_output_files, "models_used": models_used}

    except Exception as e:
        logger.error(f"Separation error: {e}")
        traceback.print_exc()
        update_status("error", 0, error=str(e))

        # Clean up on error
        output_dir = f"{STORAGE_DIR}/outputs/{task_id}"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)

        return {"task_id": task_id, "status": "error", "error": str(e), "models_used": models_used}


# --- FastAPI Application ---

class PrettyJSONResponse(StarletteResponse):
    media_type = "application/json"

    def render(self, content: typing.Any) -> bytes:
        return json.dumps(content, ensure_ascii=False, allow_nan=False, indent=4, separators=(", ", ": ")).encode("utf-8")


web_app = FastAPI(
    title="Audio Separator API",
    description="Separate vocals from instrumental tracks using AI (Cloud Run GPU)",
    version=AUDIO_SEPARATOR_VERSION,
)

web_app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@web_app.post("/separate")
async def separate_audio(
    file: UploadFile = File(..., description="Audio file to separate"),
    model: Optional[str] = Form(None, description="Single model to use for separation"),
    models: Optional[str] = Form(None, description='JSON list of models, e.g. ["model1.ckpt", "model2.onnx"]'),
    preset: Optional[str] = Form(None, description="Ensemble preset name (e.g. instrumental_clean, karaoke)"),
    # Output parameters
    output_format: str = Form("flac", description="Output format"),
    output_bitrate: Optional[str] = Form(None, description="Output bitrate"),
    normalization_threshold: float = Form(0.9),
    amplification_threshold: float = Form(0.0),
    output_single_stem: Optional[str] = Form(None),
    invert_using_spec: bool = Form(False),
    sample_rate: int = Form(44100),
    use_soundfile: bool = Form(False),
    use_autocast: bool = Form(False),
    custom_output_names: Optional[str] = Form(None),
    # MDX parameters
    mdx_segment_size: int = Form(256),
    mdx_overlap: float = Form(0.25),
    mdx_batch_size: int = Form(1),
    mdx_hop_length: int = Form(1024),
    mdx_enable_denoise: bool = Form(False),
    # VR parameters
    vr_batch_size: int = Form(1),
    vr_window_size: int = Form(512),
    vr_aggression: int = Form(5),
    vr_enable_tta: bool = Form(False),
    vr_high_end_process: bool = Form(False),
    vr_enable_post_process: bool = Form(False),
    vr_post_process_threshold: float = Form(0.2),
    # Demucs parameters
    demucs_segment_size: str = Form("Default"),
    demucs_shifts: int = Form(2),
    demucs_overlap: float = Form(0.25),
    demucs_segments_enabled: bool = Form(True),
    # MDXC parameters
    mdxc_segment_size: int = Form(256),
    mdxc_override_model_segment_size: bool = Form(False),
    mdxc_overlap: int = Form(8),
    mdxc_batch_size: int = Form(1),
    mdxc_pitch_shift: int = Form(0),
) -> dict:
    """Upload an audio file and separate it into stems."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    try:
        # Parse models parameter
        models_list = None
        if models:
            try:
                models_list = json.loads(models)
                if not isinstance(models_list, list):
                    raise ValueError("Models must be a JSON list")
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, detail=f"Invalid JSON in models parameter: {e}")
        elif model:
            models_list = [model]

        # Parse custom_output_names
        custom_output_names_dict = None
        if custom_output_names:
            try:
                custom_output_names_dict = json.loads(custom_output_names)
                if not isinstance(custom_output_names_dict, dict):
                    raise ValueError("Custom output names must be a JSON object")
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, detail=f"Invalid JSON in custom_output_names parameter: {e}")

        audio_data = await file.read()
        task_id = str(uuid.uuid4())

        # Set initial status
        job_status_store[task_id] = {
            "task_id": task_id,
            "status": "submitted",
            "progress": 0,
            "original_filename": file.filename,
            "models_used": [f"preset:{preset}"] if preset else (models_list or ["default"]),
            "total_models": 1 if preset else (len(models_list) if models_list else 1),
            "current_model_index": 0,
            "files": {},
        }

        # Run separation in a background thread to not block the event loop
        # but keep the request alive (Cloud Run keeps the instance warm)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: separate_audio_sync(
                audio_data,
                file.filename,
                task_id,
                models_list,
                preset,
                output_format,
                output_bitrate,
                normalization_threshold,
                amplification_threshold,
                output_single_stem,
                invert_using_spec,
                sample_rate,
                use_soundfile,
                use_autocast,
                custom_output_names_dict,
                mdx_segment_size,
                mdx_overlap,
                mdx_batch_size,
                mdx_hop_length,
                mdx_enable_denoise,
                vr_batch_size,
                vr_window_size,
                vr_aggression,
                vr_enable_tta,
                vr_high_end_process,
                vr_enable_post_process,
                vr_post_process_threshold,
                demucs_segment_size,
                demucs_shifts,
                demucs_overlap,
                demucs_segments_enabled,
                mdxc_segment_size,
                mdxc_override_model_segment_size,
                mdxc_overlap,
                mdxc_batch_size,
                mdxc_pitch_shift,
            ),
        )

        # Return the final status (completed or error)
        return job_status_store.get(task_id, {"task_id": task_id, "status": "error", "error": "Job lost"})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Separation failed: {str(e)}") from e


@web_app.get("/status/{task_id}")
async def get_job_status(task_id: str) -> dict:
    """Get the status of a separation job."""
    if task_id in job_status_store:
        return job_status_store[task_id]
    return {
        "task_id": task_id,
        "status": "not_found",
        "progress": 0,
        "error": "Job not found - may have been cleaned up or never existed",
    }


@web_app.get("/download/{task_id}/{file_hash}")
async def download_file(task_id: str, file_hash: str) -> Response:
    """Download a separated audio file using its hash identifier."""
    try:
        # Look up filename from job status
        status_data = job_status_store.get(task_id)
        if not status_data:
            raise HTTPException(status_code=404, detail="Task not found")

        files_dict = status_data.get("files", {})

        # Handle both dict (hash→filename) and list (legacy) formats
        actual_filename = None
        if isinstance(files_dict, dict):
            actual_filename = files_dict.get(file_hash)
        elif isinstance(files_dict, list):
            for fname in files_dict:
                if generate_file_hash(fname) == file_hash:
                    actual_filename = fname
                    break

        if not actual_filename:
            raise HTTPException(status_code=404, detail=f"File with hash {file_hash} not found")

        file_path = f"{STORAGE_DIR}/outputs/{task_id}/{actual_filename}"
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found on disk: {actual_filename}")

        with open(file_path, "rb") as f:
            file_data = f.read()

        detected_type = filetype.guess(file_data)
        content_type = detected_type.mime if detected_type and detected_type.mime else "application/octet-stream"

        ascii_filename = "".join(c if ord(c) < 128 else "_" for c in actual_filename)
        encoded_filename = quote(actual_filename, safe="")
        content_disposition = f'attachment; filename="{ascii_filename}"; filename*=UTF-8\'\'{encoded_filename}'

        return Response(content=file_data, media_type=content_type, headers={"Content-Disposition": content_disposition})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}") from e


@web_app.get("/models-json")
async def get_available_models() -> PrettyJSONResponse:
    """Get list of available separation models."""
    from audio_separator.separator import Separator

    separator = Separator(info_only=True, model_file_dir=MODEL_DIR)
    model_list = separator.list_supported_model_files()
    return PrettyJSONResponse(content=model_list)


@web_app.get("/models")
async def get_simplified_models_list(filter_sort_by: str = None) -> PlainTextResponse:
    """Get simplified model list in plain text format."""
    from audio_separator.separator import Separator

    separator = Separator(info_only=True, model_file_dir=MODEL_DIR)
    models_data = separator.get_simplified_model_list(filter_sort_by=filter_sort_by)

    if not models_data:
        return PlainTextResponse("No models found")

    filename_width = max(len("Model Filename"), max(len(f) for f in models_data.keys()))
    arch_width = max(len("Arch"), max(len(info["Type"]) for info in models_data.values()))
    stems_width = max(len("Output Stems (SDR)"), max(len(", ".join(info["Stems"])) for info in models_data.values()))
    name_width = max(len("Friendly Name"), max(len(info["Name"]) for info in models_data.values()))
    total_width = filename_width + arch_width + stems_width + name_width + 15

    output_lines = [
        "-" * total_width,
        f"{'Model Filename':<{filename_width}}  {'Arch':<{arch_width}}  {'Output Stems (SDR)':<{stems_width}}  {'Friendly Name'}",
        "-" * total_width,
    ]
    for fname, info in models_data.items():
        stems = ", ".join(info["Stems"])
        output_lines.append(f"{fname:<{filename_width}}  {info['Type']:<{arch_width}}  {stems:<{stems_width}}  {info['Name']}")

    return PlainTextResponse("\n".join(output_lines))


@web_app.get("/presets")
async def list_presets() -> PrettyJSONResponse:
    """List available ensemble presets."""
    from audio_separator.separator import Separator

    separator = Separator(info_only=True, model_file_dir=MODEL_DIR)
    presets = separator.list_ensemble_presets()
    return PrettyJSONResponse(content=presets)


@web_app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "audio-separator-api",
        "version": AUDIO_SEPARATOR_VERSION,
        "models_ready": models_ready,
        "platform": "cloud-run",
    }


@web_app.get("/")
async def root() -> dict:
    """Root endpoint with API information."""
    return {
        "message": "Audio Separator API",
        "version": AUDIO_SEPARATOR_VERSION,
        "platform": "cloud-run-gpu",
        "description": "Separate vocals from instrumental tracks using AI",
        "features": [
            "Ensemble preset support (instrumental_clean, karaoke, etc.)",
            "Multiple model processing in single job",
            "Full separator parameter compatibility",
            "GPU-accelerated processing (NVIDIA L4)",
            "All MDX, VR, Demucs, and MDXC architectures supported",
        ],
        "endpoints": {
            "POST /separate": "Upload and separate audio file (supports presets, multiple models, all parameters)",
            "GET /status/{task_id}": "Get job status and progress",
            "GET /download/{task_id}/{file_hash}": "Download separated file using hash identifier",
            "GET /presets": "List available ensemble presets",
            "GET /models-json": "List available models (JSON)",
            "GET /models": "List available models (plain text)",
            "GET /health": "Health check",
        },
    }


@web_app.on_event("startup")
async def startup_event():
    """Download models from GCS on startup."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(f"{STORAGE_DIR}/outputs", exist_ok=True)

    # Download models in background thread to not block startup probe
    thread = threading.Thread(target=download_models_from_gcs, daemon=True)
    thread.start()


if __name__ == "__main__":
    uvicorn.run(web_app, host="0.0.0.0", port=PORT)
