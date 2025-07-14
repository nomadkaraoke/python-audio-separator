"""
Audio Separator API - Simple Modal Deployment
A FastAPI service for separating vocals from instrumental tracks using audio-separator

Features:
- Asynchronous job processing
- Progress tracking and status polling
- Persistent storage for models and outputs
- GPU acceleration support
- Multiple audio format support

Usage with Remote CLI:
1. Install audio-separator package: pip install audio-separator
2. Set environment variable: export AUDIO_SEPARATOR_API_URL="https://your-deployment-url.modal.run"
3. Use the remote CLI:
   - audio-separator-remote separate song.mp3
   - audio-separator-remote separate song.mp3 --model UVR-MDX-NET-Inst_HQ_4
   - audio-separator-remote status <task_id>
   - audio-separator-remote models
   - audio-separator-remote download <task_id> <filename>
"""

# Standard library imports
import logging
import os
import shutil
import traceback
import uuid
import json
from importlib.metadata import version
import typing
from typing import Optional

# Third-party imports
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response as StarletteResponse, PlainTextResponse
import filetype
import modal

# Local imports
from audio_separator.separator import Separator

# Constants
DEFAULT_MODEL_NAME = "default"  # Used when no model is specified

# Get the version of the installed audio-separator package
try:
    AUDIO_SEPARATOR_VERSION = version("audio-separator")
except Exception:
    # Fallback version if package version cannot be determined
    AUDIO_SEPARATOR_VERSION = "unknown"

# Create Modal app
app = modal.App("audio-separator")

# Define the container image; we're using CUDA for hardware acceleration and Python 3.13 for optimal performance
image = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.13")
    .apt_install(
        [
            # Core system packages
            "curl",
            "wget",
            # Audio libraries and dependencies
            "libsndfile1",
            "libsndfile1-dev",
            "libsox-dev",
            "sox",
            "libportaudio2",
            "portaudio19-dev",
            "libasound2-dev",
            "libpulse-dev",
            "libjack-dev",
            # Sample rate conversion library
            "libsamplerate0",
            "libsamplerate0-dev",
            # Build tools for compiling Python packages with C extensions
            "build-essential",
            "clang",
            "gcc",
            "g++",
            "make",
            "cmake",
            "pkg-config",
        ]
    )
    .run_commands(
        [
            # Set up CUDA library paths for NVENC support
            "echo '/usr/local/cuda/lib64' >> /etc/ld.so.conf.d/cuda.conf",
            "ldconfig",
            # Install latest FFmpeg
            "wget https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz",
            "tar -xf ffmpeg-master-latest-linux64-gpl.tar.xz",
            "cp ffmpeg-master-latest-linux64-gpl/bin/* /usr/local/bin/",
            "chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe",
            # Verify installations and NVENC support
            "ffmpeg -version",
        ]
    )
    .pip_install(
        [
            # Core audio-separator with GPU support (this pulls in most dependencies from pyproject.toml)
            "audio-separator[gpu]>=0.34.0",
            # FastAPI and web server dependencies for Modal API deployment
            "fastapi>=0.104.0",
            "uvicorn[standard]>=0.24.0",
            "python-multipart>=0.0.6",
            # File type detection for response content type
            "filetype>=1.2.0",
        ]
    )
    .env(
        {
            "AUDIO_SEPARATOR_MODEL_DIR": "/models",
            # CUDA environment for NVENC support
            "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:$LD_LIBRARY_PATH",
            "PATH": "/usr/local/cuda/bin:$PATH",
        }
    )
)

# Create persistent volume for storing separated files
volume = modal.Volume.from_name("audio-separator-storage", create_if_missing=True)

# Create persistent volume for caching downloaded models
models_volume = modal.Volume.from_name("audio-separator-models", create_if_missing=True)

# Modal Dict for job status tracking is accessed by name in functions
# job_status_dict = modal.Dict.from_name("audio-separator-job-status", create_if_missing=True)


class PrettyJSONResponse(StarletteResponse):
    """Custom JSON response class for pretty-printing JSON"""

    media_type = "application/json"

    def render(self, content: typing.Any) -> bytes:
        return json.dumps(content, ensure_ascii=False, allow_nan=False, indent=4, separators=(", ", ": ")).encode("utf-8")


@app.function(image=image, gpu="ANY", timeout=600, volumes={"/storage": volume, "/models": models_volume}, scaledown_window=300)
def separate_audio_function(audio_data: bytes, filename: str, model_name: Optional[str] = None, task_id: Optional[str] = None) -> dict:
    """
    Separate audio into vocals and accompaniment
    """
    if task_id is None:
        task_id = str(uuid.uuid4())

    def update_job_status(status: str, progress: int = 0, error: str = None, files: list = None):
        """Update job status in Modal Dict (replaces file-based system)"""
        status_data = {"task_id": task_id, "status": status, "progress": progress, "original_filename": filename, "model_used": model_name or DEFAULT_MODEL_NAME, "files": files or []}
        if error:
            status_data["error"] = error

        # Access Modal Dict by name within function scope
        job_status = modal.Dict.from_name("audio-separator-job-status", create_if_missing=True)
        job_status[task_id] = status_data

    try:
        # Ensure storage directories exist
        os.makedirs("/storage/uploads", exist_ok=True)
        os.makedirs("/storage/outputs", exist_ok=True)

        # Ensure models directory exists
        os.makedirs("/models", exist_ok=True)

        # Update status: starting
        update_job_status("processing", 10)

        # Create output directory
        output_dir = f"/storage/outputs/{task_id}"
        os.makedirs(output_dir, exist_ok=True)

        # Write uploaded file directly to output directory with original filename
        input_file_path = os.path.join(output_dir, filename)
        with open(input_file_path, "wb") as f:
            f.write(audio_data)

        # Update status: initializing
        update_job_status("processing", 20)

        # Initialize separator with persistent model directory
        separator = Separator(log_level=logging.INFO, model_file_dir="/models", output_dir=output_dir, output_format="flac")

        # Load the model (if model_name is None, separator will use its default)
        update_job_status("processing", 30)
        if model_name:
            print(f"Loading specified model: {model_name}")
            separator.load_model(model_name)
        else:
            print(f"No model specified, using default model")
            separator.load_model()

        # Perform separation
        update_job_status("processing", 50)
        print(f"Separating audio file: {filename}")
        output_files = separator.separate(input_file_path)

        # Convert full paths to filenames - separator.separate() returns the actual output files
        update_job_status("processing", 80)

        print(f"Separation completed. Output files: {output_files}")

        # Check if separation actually produced output files
        if not output_files:
            error_msg = "Separation completed but produced no output files - this indicates a failure in the separation process"
            print(f"❌ {error_msg}")
            update_job_status("error", 0, error=error_msg)
            return {"task_id": task_id, "status": "error", "error": error_msg, "model_used": model_name or DEFAULT_MODEL_NAME, "original_filename": filename}

        # Convert full paths to filenames
        result_files = [os.path.basename(f) for f in output_files]
        print(f"Successfully separated into {len(result_files)} files: {result_files}")

        # Commit volume changes
        volume.commit()
        models_volume.commit()

        # Update status: completed
        update_job_status("completed", 100, files=result_files)

        return {"task_id": task_id, "status": "completed", "files": result_files, "model_used": model_name or DEFAULT_MODEL_NAME, "original_filename": filename}

    except FileNotFoundError as e:
        print(f"Input file not found: {str(e)}")
        traceback.print_exc()

        try:
            update_job_status("error", 0, error=f"Input file not found: {str(e)}")
        except Exception as status_error:
            print(f"WARNING: Failed to update job status: {status_error}")

        return {"task_id": task_id, "status": "error", "error": f"Input file not found: {str(e)}", "model_used": model_name or DEFAULT_MODEL_NAME, "original_filename": filename}

    except ValueError as e:
        print(f"Invalid input or configuration: {str(e)}")
        traceback.print_exc()

        try:
            update_job_status("error", 0, error=f"Invalid input: {str(e)}")
        except Exception as status_error:
            print(f"WARNING: Failed to update job status: {status_error}")

        return {"task_id": task_id, "status": "error", "error": f"Invalid input: {str(e)}", "model_used": DEFAULT_MODEL_NAME, "original_filename": filename}

    except Exception as e:
        print(f"Unexpected error during separation: {str(e)}")
        traceback.print_exc()

        # Update status: error
        try:
            update_job_status("error", 0, error=str(e))
        except Exception as status_error:
            print(f"WARNING: Failed to update job status: {status_error}")
            # Continue with the original error handling

        # Clean up on error
        if os.path.exists(input_file_path):
            os.unlink(input_file_path)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)

        return {"task_id": task_id, "status": "error", "error": str(e), "model_used": model_name or DEFAULT_MODEL_NAME, "original_filename": filename}


@app.function(image=image, timeout=300, volumes={"/storage": volume})
def get_job_status_function(task_id: str) -> dict:
    """
    Get the status of a separation job
    """
    try:
        # Access Modal Dict by name within function scope
        job_status = modal.Dict.from_name("audio-separator-job-status", create_if_missing=True)

        if task_id in job_status:
            return job_status[task_id]
        else:
            # Job not found - might be initializing or doesn't exist
            return {"task_id": task_id, "status": "not_found", "progress": 0, "error": "Job not found - may have been cleaned up or never existed"}
    except Exception as e:
        print(f"ERROR: Failed to access job status for {task_id}: {str(e)}")
        return {"task_id": task_id, "status": "error", "error": f"Failed to read status: {str(e)}"}


@app.function(image=image, timeout=300, volumes={"/storage": volume})
def get_file_function(task_id: str, filename: str) -> bytes:
    """
    Retrieve a separated audio file
    """
    file_path = f"/storage/outputs/{task_id}/{filename}"

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {filename}")

    with open(file_path, "rb") as f:
        return f.read()


@app.function(image=image, timeout=60, volumes={"/models": models_volume})
def list_available_models() -> dict:
    """
    List available separation models using the same approach as CLI
    """
    # Use the persistent model directory
    model_dir = "/models"

    # Ensure the model directory exists
    os.makedirs(model_dir, exist_ok=True)

    # Use the same approach as the CLI: create separator with info_only=True
    separator = Separator(info_only=True, model_file_dir=model_dir)

    # Get the list of supported models
    model_list = separator.list_supported_model_files()

    # Return the full model dictionary
    return model_list


@app.function(image=image, timeout=60, volumes={"/models": models_volume})
def get_simplified_models(filter_sort_by: str = None) -> dict:
    """
    Get simplified model list using the same approach as CLI --list_models
    """
    # Use the persistent model directory
    model_dir = "/models"

    # Ensure the model directory exists
    os.makedirs(model_dir, exist_ok=True)

    # Use the same approach as the CLI: create separator with info_only=True
    separator = Separator(info_only=True, model_file_dir=model_dir)

    # Get the simplified model list
    simplified_models = separator.get_simplified_model_list(filter_sort_by=filter_sort_by)

    return simplified_models


web_app = FastAPI(title="Audio Separator API", description="Separate vocals from instrumental tracks using AI", version=AUDIO_SEPARATOR_VERSION)

web_app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@web_app.post("/separate")
async def separate_audio(
    file: UploadFile = File(..., description="Audio file to separate"),
    model: Optional[str] = Form(None, description="Model to use for separation (optional, uses audio-separator default if not specified)"),
) -> dict:
    """
    Upload an audio file and separate it into vocals and accompaniment (asynchronous processing)
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Let audio-separator handle file type validation - it supports many formats

    try:
        # Read file data
        audio_data = await file.read()

        # Generate task ID
        task_id = str(uuid.uuid4())

        # Create initial status in Modal Dict
        initial_status = {"task_id": task_id, "status": "submitted", "progress": 0, "original_filename": file.filename, "model_used": model or DEFAULT_MODEL_NAME, "files": []}
        # Access Modal Dict by name to ensure proper scope
        job_status = modal.Dict.from_name("audio-separator-job-status", create_if_missing=True)
        job_status[task_id] = initial_status

        # Submit job asynchronously
        separate_audio_function.spawn(audio_data, file.filename, model, task_id)
        return {
            "task_id": task_id,
            "status": "submitted",
            "message": "Job submitted for processing. Use /status/{task_id} to check progress.",
            "model_used": model or DEFAULT_MODEL_NAME,
            "original_filename": file.filename,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Separation failed: {str(e)}") from e


@web_app.get("/status/{task_id}")
async def get_job_status(task_id: str) -> dict:
    """
    Get the status of a separation job
    """
    try:
        status_data = get_job_status_function.remote(task_id)
        return status_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}") from e


@web_app.get("/download/{task_id}/{filename}")
async def download_file(task_id: str, filename: str) -> Response:
    """
    Download a separated audio file
    """
    try:
        file_data = get_file_function.remote(task_id, filename)

        # Detect file type from content
        detected_type = filetype.guess(file_data)

        if detected_type and detected_type.mime:
            content_type = detected_type.mime
        else:
            # Log when we can't detect the file type
            print(f"WARNING: Could not detect MIME type for {filename}, using generic type")
            content_type = "application/octet-stream"

        return Response(content=file_data, media_type=content_type, headers={"Content-Disposition": f"attachment; filename={filename}"})

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="File not found") from exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}") from e


@web_app.get("/models-json")
async def get_available_models() -> PrettyJSONResponse:
    """
    Get list of available separation models
    """
    models = list_available_models.remote()

    # Return pretty-printed JSON for better readability
    return PrettyJSONResponse(content=models)


@web_app.get("/models")
async def get_simplified_models_list(filter_sort_by: str = None) -> PlainTextResponse:
    """
    Get simplified model list in plain text format (like CLI --list_models)
    """
    models = get_simplified_models.remote(filter_sort_by=filter_sort_by)

    if not models:
        return PlainTextResponse("No models found")

    # Calculate maximum widths for each column
    filename_width = max(len("Model Filename"), max(len(filename) for filename in models.keys()))
    arch_width = max(len("Arch"), max(len(info["Type"]) for info in models.values()))
    stems_width = max(len("Output Stems (SDR)"), max(len(", ".join(info["Stems"])) for info in models.values()))
    name_width = max(len("Friendly Name"), max(len(info["Name"]) for info in models.values()))

    # Calculate total width for separator line
    total_width = filename_width + arch_width + stems_width + name_width + 15  # 15 accounts for spacing between columns

    # Format the output with dynamic widths and extra spacing
    output_lines = []
    output_lines.append("-" * total_width)
    output_lines.append(f"{'Model Filename':<{filename_width}}  {'Arch':<{arch_width}}  {'Output Stems (SDR)':<{stems_width}}  {'Friendly Name'}")
    output_lines.append("-" * total_width)

    for filename, info in models.items():
        stems = ", ".join(info["Stems"])
        output_lines.append(f"{filename:<{filename_width}}  {info['Type']:<{arch_width}}  {stems:<{stems_width}}  {info['Name']}")

    return PlainTextResponse("\n".join(output_lines))


@web_app.get("/health")
async def health_check() -> dict:
    """
    Health check endpoint
    """
    return {"status": "healthy", "service": "audio-separator-api", "version": AUDIO_SEPARATOR_VERSION}


@web_app.get("/")
async def root() -> dict:
    """
    Root endpoint with API information
    """
    return {
        "message": "Audio Separator API",
        "version": AUDIO_SEPARATOR_VERSION,
        "description": ("Separate vocals from instrumental tracks using AI - " "supports all formats that audio-separator CLI supports"),
        "endpoints": {
            "POST /separate": "Upload and separate audio file",
            "GET /status/{task_id}": "Get job status and progress",
            "GET /download/{task_id}/{filename}": "Download separated file",
            "GET /models-json": "List available models (JSON format)",
            "GET /models": "List available models (plain text format like CLI --list_models)",
            "GET /health": "Health check",
        },
        "note": ("This is a minimal wrapper around audio-separator - " "file types and sizes are handled by the underlying library"),
        "remote_cli": {
            "install": "pip install audio-separator",
            "setup": 'export AUDIO_SEPARATOR_API_URL="https://your-deployment-url.modal.run"',
            "usage": [
                "audio-separator-remote separate song.mp3",
                "audio-separator-remote separate song.mp3 --model UVR-MDX-NET-Inst_HQ_4",
                "audio-separator-remote status <task_id>",
                "audio-separator-remote models",
            ],
        },
    }


@app.function(image=image, timeout=600, scaledown_window=300, volumes={"/storage": volume})
@modal.asgi_app()
def api() -> FastAPI:
    """
    Deploy the FastAPI app as a Modal ASGI application
    """
    return web_app
