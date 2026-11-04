"""MDX stems keep the input's level whatever gain normalization applied before demixing (#318)."""

import logging
from unittest.mock import Mock

import numpy as np
import pytest

from audio_separator.separator.architectures.mdx_separator import MDXSeparator

VOCAL_SHARE = 0.25


def mdx_double(mix, amplification_threshold):
    separator = Mock(spec=MDXSeparator)
    separator.logger = logging.getLogger(__name__)
    separator.normalization_threshold = 0.9
    separator.amplification_threshold = amplification_threshold
    separator.output_single_stem = None
    separator.prepare_mix.return_value = mix
    # A linear stand-in for the model: the "vocals" are a fixed share of whatever mix it is given.
    separator.demix.side_effect = lambda wave, is_match_mix=False: VOCAL_SHARE * wave
    separator.primary_source = None
    separator.secondary_source = None
    separator.primary_stem_name = "Vocals"
    separator.secondary_stem_name = "Instrumental"
    separator.compensate = 1.0
    separator.get_stem_output_path.side_effect = ["input_(Instrumental).wav", "input_(Vocals).wav"]
    return separator


@pytest.mark.parametrize(
    "peak, amplification_threshold",
    [
        (0.4, 0.0),  # between the thresholds: normalize() leaves the mix alone
        (1.5, 0.0),  # above normalization_threshold: scaled down before demixing
        (0.05, 0.1),  # below amplification_threshold: scaled up before demixing
    ],
)
@pytest.mark.parametrize("invert_using_spec", [False, True])
def test_mdx_stems_keep_input_level(peak, amplification_threshold, invert_using_spec, monkeypatch):
    rng = np.random.default_rng(318)
    mix = rng.uniform(-1, 1, (2, 64)).astype(np.float32)
    mix *= peak / np.abs(mix).max()
    original = mix.copy()

    separator = mdx_double(mix, amplification_threshold)
    separator.invert_using_spec = invert_using_spec
    # Spectrogram inversion is linear for these inputs; subtracting is enough to check the scale it is given.
    monkeypatch.setattr("audio_separator.separator.architectures.mdx_separator.spec_utils.invert_stem", lambda raw_mix, stem: raw_mix.T * (1 / VOCAL_SHARE) - stem)

    MDXSeparator.separate(separator, "input.wav")

    np.testing.assert_allclose(separator.primary_source, VOCAL_SHARE * original.T, rtol=1e-5)
    np.testing.assert_allclose(separator.secondary_source, (1 - VOCAL_SHARE) * original.T, rtol=1e-5)
    np.testing.assert_allclose(separator.primary_source + separator.secondary_source, original.T, rtol=1e-5)
