import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from audio_separator.separator import Separator
from audio_separator.separator.architectures.vr_separator import VRSeparator


@pytest.fixture
def separator(tmp_path):
    return Separator(model_file_dir=tmp_path / "models", output_dir=tmp_path / "output", info_only=True)


def test_load_model_reuses_matching_instance(separator):
    loaded_instance = object()
    separator.model_instance = loaded_instance
    separator._loaded_model_filename = "model.ckpt"

    with patch.object(separator, "download_model_files") as download_model_files:
        separator.load_model("model.ckpt")

    download_model_files.assert_not_called()
    assert separator.model_instance is loaded_instance
    assert separator.model_filename == "model.ckpt"
    assert separator.model_filenames == ["model.ckpt"]


def test_load_model_normalizes_single_item_list_before_reuse(separator):
    separator.model_instance = object()
    separator._loaded_model_filename = "model.ckpt"

    with patch.object(separator, "download_model_files") as download_model_files:
        separator.load_model(["model.ckpt"])

    download_model_files.assert_not_called()
    assert separator.model_filename == "model.ckpt"
    assert separator.model_filenames == ["model.ckpt"]


def test_load_model_reuse_restores_loaded_model_metadata(separator):
    loaded_instance = object()
    separator.model_instance = loaded_instance
    separator._loaded_model_filename = "first.ckpt"
    separator._loaded_model_friendly_name = "First model"
    separator._loaded_model_is_uvr_vip = False

    def download_second_model(_model_filename):
        separator.model_friendly_name = "Second VIP model"
        separator.model_is_uvr_vip = True
        return "second.ckpt", "MDXC", "Second VIP model", "/tmp/second.ckpt", None

    with (
        patch.object(separator, "download_model_files", side_effect=download_second_model) as download_model_files,
        patch.object(separator, "load_model_data_using_hash", return_value={}),
    ):
        separator.download_model_and_data("second.ckpt")
        separator.load_model("first.ckpt")

    download_model_files.assert_called_once_with("second.ckpt")
    assert separator.model_instance is loaded_instance
    assert separator.model_friendly_name == "First model"
    assert separator.model_is_uvr_vip is False


def test_load_model_force_reload_bypasses_reuse(separator):
    separator.model_instance = object()
    separator._loaded_model_filename = "model.ckpt"

    with patch.object(separator, "download_model_files", side_effect=RuntimeError("reload attempted")) as download_model_files:
        with pytest.raises(RuntimeError, match="reload attempted"):
            separator.load_model("model.ckpt", force_reload=True)

    download_model_files.assert_called_once_with("model.ckpt")


def test_load_model_reloads_when_instance_is_missing(separator):
    separator._loaded_model_filename = "model.ckpt"

    with patch.object(separator, "download_model_files", side_effect=RuntimeError("reload attempted")):
        with pytest.raises(RuntimeError, match="reload attempted"):
            separator.load_model("model.ckpt")


def test_load_model_reloads_different_model_without_poisoning_cache(separator):
    loaded_instance = object()
    separator.model_instance = loaded_instance
    separator.model_filename = "first.ckpt"
    separator.model_filenames = ["first.ckpt"]
    separator._loaded_model_filename = "first.ckpt"
    separator.model_friendly_name = "First model"
    separator.model_is_uvr_vip = False

    def fail_download(_model_filename):
        separator.model_friendly_name = "Second model"
        separator.model_is_uvr_vip = True
        raise RuntimeError("load failed")

    with patch.object(separator, "download_model_files", side_effect=fail_download):
        with pytest.raises(RuntimeError, match="load failed"):
            separator.load_model("second.ckpt")

    assert separator.model_instance is loaded_instance
    assert separator.model_filename == "first.ckpt"
    assert separator.model_filenames == ["first.ckpt"]
    assert separator._loaded_model_filename == "first.ckpt"
    assert separator.model_friendly_name == "First model"
    assert separator.model_is_uvr_vip is False


@pytest.mark.parametrize(
    "failure",
    [RuntimeError("construction failed"), SystemExit("construction stopped")],
    ids=["runtime-error", "system-exit"],
)
def test_load_model_constructor_failure_preserves_previous_selection(separator, failure):
    loaded_instance = object()
    separator.model_instance = loaded_instance
    separator.model_filename = "first.ckpt"
    separator.model_filenames = ["first.ckpt"]
    separator._loaded_model_filename = "first.ckpt"
    separator.model_friendly_name = "First model"
    separator.model_is_uvr_vip = False
    separator_class = MagicMock(side_effect=failure)
    architecture_module = SimpleNamespace(MDXCSeparator=separator_class)

    def resolve_download(_model_filename):
        separator.model_friendly_name = "Second model"
        separator.model_is_uvr_vip = True
        return "second.ckpt", "MDXC", "Second model", "/tmp/second.ckpt", None

    with (
        patch.object(separator, "download_model_files", side_effect=resolve_download),
        patch.object(separator, "load_model_data_using_hash", return_value={}),
        patch("audio_separator.separator.separator.importlib.import_module", return_value=architecture_module),
    ):
        with pytest.raises(type(failure), match=str(failure)):
            separator.load_model("second.ckpt")

    assert separator.model_instance is loaded_instance
    assert separator.model_filename == "first.ckpt"
    assert separator.model_filenames == ["first.ckpt"]
    assert separator._loaded_model_filename == "first.ckpt"
    assert separator.model_friendly_name == "First model"
    assert separator.model_is_uvr_vip is False


def test_load_model_policy_failure_preserves_previous_ensemble_selection(separator):
    ensemble_models = ["first.ckpt", "third.ckpt"]
    loaded_instance = object()
    separator.model_instance = loaded_instance
    separator.model_filename = list(ensemble_models)
    separator.model_filenames = list(ensemble_models)
    separator._loaded_model_filename = "first.ckpt"
    separator.model_friendly_name = "First model"
    separator.model_is_uvr_vip = False
    resolve_execution_policy = MagicMock(side_effect=RuntimeError("policy failed"))
    candidate_instance = SimpleNamespace(
        _execution_policy_resolved=False,
        is_roformer_model=False,
        resolve_execution_policy=resolve_execution_policy,
    )
    separator_class = MagicMock(return_value=candidate_instance)
    architecture_module = SimpleNamespace(MDXCSeparator=separator_class)

    with (
        patch.object(
            separator,
            "download_model_files",
            return_value=("second.ckpt", "MDXC", "Second model", "/tmp/second.ckpt", None),
        ),
        patch.object(separator, "load_model_data_using_hash", return_value={}),
        patch("audio_separator.separator.separator.importlib.import_module", return_value=architecture_module),
    ):
        with pytest.raises(RuntimeError, match="policy failed"):
            separator.load_model("second.ckpt")

    resolve_execution_policy.assert_called_once_with("mdxc")
    assert separator.model_instance is loaded_instance
    assert separator.model_filename == ensemble_models
    assert separator.model_filenames == ensemble_models
    assert separator.model_filename is not ensemble_models
    assert separator._loaded_model_filename == "first.ckpt"
    assert separator.model_friendly_name == "First model"
    assert separator.model_is_uvr_vip is False


def test_successful_load_populates_cache_for_the_next_call(separator):
    loaded_instance = object()
    separator_class = MagicMock(return_value=loaded_instance)
    architecture_module = SimpleNamespace(MDXCSeparator=separator_class)

    with (
        patch.object(
            separator,
            "download_model_files",
            return_value=("model.ckpt", "MDXC", "Model", "/tmp/model.ckpt", None),
        ) as download_model_files,
        patch.object(separator, "load_model_data_using_hash", return_value={}),
        patch("audio_separator.separator.separator.importlib.import_module", return_value=architecture_module),
    ):
        separator.load_model("model.ckpt")
        separator.load_model("model.ckpt")

    download_model_files.assert_called_once_with("model.ckpt")
    separator_class.assert_called_once()
    assert separator.model_instance is loaded_instance
    assert separator._loaded_model_filename == "model.ckpt"
    assert separator._loaded_model_friendly_name == "Model"
    assert separator._loaded_model_is_uvr_vip is False


def test_force_reload_repeats_a_successful_load(separator):
    separator_class = MagicMock(side_effect=[object(), object()])
    architecture_module = SimpleNamespace(MDXCSeparator=separator_class)

    with (
        patch.object(
            separator,
            "download_model_files",
            return_value=("model.ckpt", "MDXC", "Model", "/tmp/model.ckpt", None),
        ) as download_model_files,
        patch.object(separator, "load_model_data_using_hash", return_value={}),
        patch("audio_separator.separator.separator.importlib.import_module", return_value=architecture_module),
    ):
        separator.load_model("model.ckpt")
        first_instance = separator.model_instance
        separator.load_model("model.ckpt", force_reload=True)

    assert download_model_files.call_count == 2
    assert separator_class.call_count == 2
    assert separator.model_instance is not first_instance
    assert separator._loaded_model_filename == "model.ckpt"


def test_load_model_preserves_multi_model_ensemble_behavior(separator):
    models = ["first.ckpt", "second.ckpt"]

    with patch.object(separator, "download_model_files") as download_model_files:
        separator.load_model(models)

    download_model_files.assert_not_called()
    assert separator.model_filename == models
    assert separator.model_filenames == models
    assert separator.model_filename is not models


def test_load_model_expands_ensemble_preset_before_reuse(separator):
    preset_models = ["first.ckpt", "second.ckpt"]
    separator._ensemble_preset_models = preset_models

    with patch.object(separator, "download_model_files") as download_model_files:
        separator.load_model()

    download_model_files.assert_not_called()
    assert separator.model_filename == preset_models
    assert separator.model_filenames == preset_models


def _make_vr_separator(model_run):
    separator = object.__new__(VRSeparator)
    separator.logger = logging.getLogger(__name__)
    separator.model_run = model_run
    separator.model_params = SimpleNamespace(param={"bins": 128})
    separator.model_capacity = (32, 128)
    separator.is_vr_51_model = False
    separator.model_path = "/tmp/model.pth"
    separator.torch_device = torch.device("cpu")
    return separator


def test_vr_model_reuses_loaded_torch_module():
    loaded_model = torch.nn.Linear(2, 2)
    separator = _make_vr_separator(loaded_model)

    with (
        patch("audio_separator.separator.architectures.vr_separator.nets.determine_model_capacity") as determine_model_capacity,
        patch("audio_separator.separator.architectures.vr_separator.torch.load") as load_weights,
    ):
        separator._ensure_model_loaded(31191)

    determine_model_capacity.assert_not_called()
    load_weights.assert_not_called()
    assert separator.model_run is loaded_model


def test_vr_model_loads_placeholder_once():
    separator = _make_vr_separator(lambda: None)
    loaded_model = MagicMock(spec=torch.nn.Module)
    state_dict = {"weight": torch.tensor([1.0])}

    with (
        patch(
            "audio_separator.separator.architectures.vr_separator.nets.determine_model_capacity",
            return_value=loaded_model,
        ) as determine_model_capacity,
        patch(
            "audio_separator.separator.architectures.vr_separator.torch.load",
            return_value=state_dict,
        ) as load_weights,
    ):
        separator._ensure_model_loaded(31191)

    determine_model_capacity.assert_called_once_with(256, 31191)
    load_weights.assert_called_once_with("/tmp/model.pth", map_location="cpu")
    loaded_model.load_state_dict.assert_called_once_with(state_dict)
    loaded_model.to.assert_called_once_with(torch.device("cpu"))
    assert separator.model_run is loaded_model


def test_vr_model_retries_after_weight_loading_failure():
    placeholder = lambda: None
    separator = _make_vr_separator(placeholder)
    failed_model = MagicMock(spec=torch.nn.Module)
    failed_model.load_state_dict.side_effect = RuntimeError("invalid weights")
    loaded_model = MagicMock(spec=torch.nn.Module)

    with (
        patch(
            "audio_separator.separator.architectures.vr_separator.nets.determine_model_capacity",
            side_effect=[failed_model, loaded_model],
        ) as determine_model_capacity,
        patch(
            "audio_separator.separator.architectures.vr_separator.torch.load",
            side_effect=[{"bad": torch.tensor([1.0])}, {"weight": torch.tensor([2.0])}],
        ) as load_weights,
    ):
        with pytest.raises(RuntimeError, match="invalid weights"):
            separator._ensure_model_loaded(31191)

        assert separator.model_run is placeholder
        separator._ensure_model_loaded(31191)

    assert determine_model_capacity.call_count == 2
    assert load_weights.call_count == 2
    loaded_model.load_state_dict.assert_called_once()
    loaded_model.to.assert_called_once_with(torch.device("cpu"))
    assert separator.model_run is loaded_model


def test_effective_mode_is_reset_while_an_ensemble_is_selected(separator):
    separator.model_instance = SimpleNamespace(effective_precision="native_fp16", effective_torch_compile=True)
    separator._loaded_model_filename = "first.ckpt"

    separator.load_model(["first.ckpt", "second.ckpt"])

    assert separator.effective_precision == "fp32"
    assert separator.effective_torch_compile is False


def test_failed_separation_clears_reused_instance_state(separator):
    model_instance = MagicMock()
    model_instance.effective_precision = "fp32"
    model_instance.torch_device = torch.device("cpu")
    model_instance.separate.side_effect = RuntimeError("inference failed")
    model_instance.clear_gpu_cache.side_effect = RuntimeError("cleanup failed")
    separator.model_instance = model_instance

    with pytest.raises(RuntimeError, match="inference failed"):
        separator._separate_file("input.wav")

    model_instance.clear_gpu_cache.assert_called_once_with()
    model_instance.clear_file_specific_paths.assert_called_once_with()
