"""Model downloads come from our mirror first and must match their published SHA-256."""

import hashlib
import json
from importlib import resources
from unittest.mock import MagicMock, patch

import pytest

from audio_separator.separator import Separator

MIRROR = "https://github.com/nomadkaraoke/python-audio-separator/releases/download/model-configs"
UVR = "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models"
VIP = "https://github.com/Anjok0109/ai_magic/releases/download/v5"

GOOD = b"genuine model bytes"
BAD = b"tampered model bytes"


@pytest.fixture
def separator(tmp_path, monkeypatch):
    # The env var would redirect downloads into a real model directory.
    monkeypatch.delenv("AUDIO_SEPARATOR_MODEL_DIR", raising=False)
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    sep = Separator(model_file_dir=str(model_dir), output_dir=str(tmp_path / "output"), info_only=True)
    sep._model_checksums = {"model.ckpt": hashlib.sha256(GOOD).hexdigest(), "config.yaml": hashlib.sha256(b"cfg").hexdigest()}
    return sep


def _serve(content_by_url):
    """Fake requests.get: URLs in content_by_url return 200 with those bytes, anything else 404."""

    def fake_get(url, stream=True, timeout=None):
        response = MagicMock()
        if url in content_by_url:
            response.status_code = 200
            response.headers = {"content-length": str(len(content_by_url[url]))}
            response.iter_content.return_value = [content_by_url[url]]
        else:
            response.status_code = 404
        return response

    return patch("audio_separator.separator.separator.requests.get", side_effect=fake_get)


def _supported(model_type, friendly_name, download_files):
    return {model_type: {friendly_name: {"filename": download_files[0], "download_files": download_files}}}


def test_model_is_fetched_from_mirror_before_upstream(separator):
    with patch.object(separator, "list_supported_model_files", return_value=_supported("MDXC", "Roformer Model: X", ["model.ckpt"])), \
         _serve({f"{MIRROR}/model.ckpt": GOOD, f"{UVR}/model.ckpt": GOOD}) as get:
        separator.download_model_files("model.ckpt")

    assert get.call_args_list[0].args[0] == f"{MIRROR}/model.ckpt"
    assert get.call_count == 1


def test_upstream_is_used_when_mirror_lacks_the_file(separator, tmp_path):
    with patch.object(separator, "list_supported_model_files", return_value=_supported("MDXC", "Roformer Model: X", ["model.ckpt"])), \
         _serve({f"{UVR}/model.ckpt": GOOD}):
        separator.download_model_files("model.ckpt")

    assert (tmp_path / "models" / "model.ckpt").read_bytes() == GOOD


def test_vip_models_still_come_from_vip_repo_first(separator):
    with patch.object(separator, "list_supported_model_files", return_value=_supported("MDX", "MDX-Net Model VIP: X", ["model.ckpt"])), \
         _serve({f"{VIP}/model.ckpt": GOOD, f"{MIRROR}/model.ckpt": GOOD}) as get:
        separator.download_model_files("model.ckpt")

    assert get.call_args_list[0].args[0] == f"{VIP}/model.ckpt"


def test_mdxc_yaml_config_comes_from_mirror_first(separator):
    files = ["model.ckpt", "config.yaml"]
    with patch.object(separator, "list_supported_model_files", return_value=_supported("MDXC", "Roformer Model: X", files)), \
         _serve({f"{MIRROR}/model.ckpt": GOOD, f"{MIRROR}/config.yaml": b"cfg"}) as get:
        result = separator.download_model_files("model.ckpt")

    assert [c.args[0] for c in get.call_args_list] == [f"{MIRROR}/model.ckpt", f"{MIRROR}/config.yaml"]
    assert result[4] == "config.yaml"


def test_tampered_download_is_rejected_and_next_source_used(separator, tmp_path):
    with patch.object(separator, "list_supported_model_files", return_value=_supported("MDXC", "Roformer Model: X", ["model.ckpt"])), \
         _serve({f"{MIRROR}/model.ckpt": BAD, f"{UVR}/model.ckpt": GOOD}):
        separator.download_model_files("model.ckpt")

    assert (tmp_path / "models" / "model.ckpt").read_bytes() == GOOD
    assert not (tmp_path / "models" / "model.ckpt.partial").exists()


def test_download_fails_when_every_source_is_tampered(separator, tmp_path):
    with patch.object(separator, "list_supported_model_files", return_value=_supported("MDXC", "Roformer Model: X", ["model.ckpt"])), \
         _serve({f"{MIRROR}/model.ckpt": BAD, f"{UVR}/model.ckpt": BAD}):
        with pytest.raises(RuntimeError, match="checksum"):
            separator.download_model_files("model.ckpt")

    assert list((tmp_path / "models").iterdir()) == []


def test_tampered_cached_file_is_replaced(separator, tmp_path):
    (tmp_path / "models" / "model.ckpt").write_bytes(BAD)

    with _serve({f"{MIRROR}/model.ckpt": GOOD}) as get:
        separator.download_file_from_first_available([f"{MIRROR}/model.ckpt"], str(tmp_path / "models" / "model.ckpt"))

    assert get.call_count == 1
    assert (tmp_path / "models" / "model.ckpt").read_bytes() == GOOD


def test_verified_cached_file_is_not_downloaded_or_rehashed(separator, tmp_path):
    path = tmp_path / "models" / "model.ckpt"
    path.write_bytes(GOOD)

    with _serve({}) as get:
        separator.download_file_if_not_exists(f"{MIRROR}/model.ckpt", str(path), verify_checksum=True)
        with patch("audio_separator.separator.separator.hashlib.sha256") as sha256:
            separator.download_file_if_not_exists(f"{MIRROR}/model.ckpt", str(path), verify_checksum=True)

    get.assert_not_called()
    sha256.assert_not_called()


def test_files_without_a_published_checksum_are_accepted(separator, tmp_path):
    with _serve({f"{MIRROR}/new_model.ckpt": b"anything"}):
        separator.download_file_if_not_exists(f"{MIRROR}/new_model.ckpt", str(tmp_path / "models" / "new_model.ckpt"), verify_checksum=True)

    assert (tmp_path / "models" / "new_model.ckpt").read_bytes() == b"anything"


def test_metadata_downloads_are_not_checked_against_model_checksums(separator, tmp_path):
    # Mutable lists like download_checks.json share names with old snapshots in the mirror release.
    separator._model_checksums["download_checks.json"] = hashlib.sha256(b"old snapshot").hexdigest()

    with _serve({"https://example.com/download_checks.json": b"current list"}):
        separator.download_file_if_not_exists("https://example.com/download_checks.json", str(tmp_path / "models" / "download_checks.json"))

    assert (tmp_path / "models" / "download_checks.json").read_bytes() == b"current list"


def test_bundled_manifest_covers_default_models():
    with resources.open_text("audio_separator", "model-checksums.json") as f:
        checksums = json.load(f)["sha256"]

    for filename in ["model_bs_roformer_ep_317_sdr_12.9755.ckpt", "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt", "htdemucs_ft.yaml", "f7e0c4bc-ba3fe64a.th"]:
        assert len(checksums[filename]) == 64
