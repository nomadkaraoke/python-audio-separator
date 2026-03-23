"""Download ensemble preset models for baking into Docker image."""
import json
import importlib.resources as resources
from audio_separator.separator import Separator

with resources.open_text("audio_separator", "ensemble_presets.json") as f:
    presets = json.load(f)["presets"]

models_to_download = set()
for preset_name in ["instrumental_clean", "karaoke"]:
    models_to_download.update(presets[preset_name]["models"])

print(f"Downloading {len(models_to_download)} models for ensemble presets...")
for model in sorted(models_to_download):
    print(f"  Downloading: {model}")
    sep = Separator(model_file_dir="/models")
    sep.load_model(model)
    print(f"  Done: {model}")
print("All models downloaded successfully.")
