"""
Forward-pass budget for RoFormer inference on real model geometries.

The RoFormer cost of a separation is (number of model forward passes) x (cost per pass).
v0.47.0 (#299) changed how `overlap` maps to the chunk step, which silently made the
karaoke models (YAML num_overlap 4/8) run 4-6x more forward passes than v0.44.2 for the
same track. Nothing failed; downstream users just saw separations take 3-6x longer.

These tests pin the pass count for a 3-minute stereo track. If a change to chunking,
step or tail handling moves any of these numbers, that is a deliberate compute change:
update the budget AND call it out in the release notes.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from ml_collections import ConfigDict

from audio_separator.separator.architectures.mdxc_separator import MDXCSeparator

SAMPLE_RATE = 44100
TRACK_SECONDS = 180


class CountingIdentityModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.calls = 0

    def forward(self, audio):
        self.calls += 1
        return audio.unsqueeze(1)


def _count_forward_passes(dim_t: int, overlap: int) -> int:
    separator = MDXCSeparator.__new__(MDXCSeparator)
    separator.pitch_shift = 0
    separator.is_roformer = True
    separator.override_model_segment_size = False
    separator.model_data_cfgdict = ConfigDict(
        {
            "inference": {"dim_t": dim_t},
            "training": {"target_instrument": "Vocals", "instruments": ["Vocals"]},
            "model": {"stft_hop_length": 441},
            "audio": {"hop_length": 441, "sample_rate": SAMPLE_RATE},
        }
    )
    separator.overlap = overlap
    separator.model_run = CountingIdentityModel()
    separator.is_primary_stem_main_target = False
    separator.logger = Mock()

    mix = np.zeros((2, SAMPLE_RATE * TRACK_SECONDS), dtype=np.float32)
    with patch("audio_separator.separator.architectures.mdxc_separator.tqdm", side_effect=lambda values: values):
        separator.demix(mix)
    return separator.model_run.calls


# (model, inference.dim_t, YAML num_overlap, passes at YAML num_overlap, passes at overlap=2)
# v0.44.2 ran every one of these at 23 passes (fixed 8 s step, clamped to chunk size).
MODEL_BUDGETS = [
    ("mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956", 801, 4, 87, 44),
    ("mel_band_roformer_karaoke_gabox_v2", 1101, 8, 124, 32),
    ("mel_band_roformer_karaoke_becruily", 1101, 8, 124, 32),
    ("mel_band_roformer_instrumental_fv7z_gabox", 1101, 2, 32, 32),
    ("bs_roformer_instrumental_resurrection_unwa", 1700, 2, 21, 21),
]


@pytest.mark.parametrize("model,dim_t,num_overlap,yaml_passes,overlap2_passes", MODEL_BUDGETS)
def test_roformer_forward_pass_budget(model, dim_t, num_overlap, yaml_passes, overlap2_passes):
    assert _count_forward_passes(dim_t, num_overlap) == yaml_passes, (
        f"{model}: forward passes at YAML num_overlap={num_overlap} changed. "
        "This changes separation time proportionally; update the budget and the release notes."
    )
    assert _count_forward_passes(dim_t, 2) == overlap2_passes, (
        f"{model}: forward passes at overlap=2 changed. "
        "This changes separation time proportionally; update the budget and the release notes."
    )


def test_pass_count_scales_with_overlap():
    """Doubling overlap roughly doubles compute: the step is chunk_size // overlap."""
    passes = [_count_forward_passes(801, overlap) for overlap in (1, 2, 4, 8)]
    assert passes == sorted(passes)
    for lower, higher in zip(passes, passes[1:]):
        assert 1.7 <= higher / lower <= 2.1
