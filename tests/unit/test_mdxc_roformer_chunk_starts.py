from unittest.mock import Mock

import pytest

from audio_separator.separator.architectures.mdxc_separator import MDXCSeparator


@pytest.mark.parametrize(
    ("audio_length", "chunk_size", "step", "expected"),
    [
        (0, 4, 2, []),
        (3, 4, 2, [0]),
        (4, 4, 2, [0]),
        (10, 4, 3, [0, 3, 6]),
        (11, 4, 3, [0, 3, 6, 7]),
        (20, 8, 5, [0, 5, 10, 12]),
        (20, 8, 2, [0, 2, 4, 6, 8, 10, 12]),
    ],
)
def test_roformer_chunk_starts_cover_tail_once(audio_length, chunk_size, step, expected):
    assert MDXCSeparator._roformer_chunk_starts(audio_length, chunk_size, step) == expected


@pytest.mark.parametrize(("chunk_size", "step"), [(0, 1), (-1, 1), (4, 0), (4, -1), (4, 5)])
def test_roformer_chunk_starts_reject_invalid_chunk_schedule(chunk_size, step):
    with pytest.raises(ValueError):
        MDXCSeparator._roformer_chunk_starts(audio_length=10, chunk_size=chunk_size, step=step)


def test_roformer_chunk_starts_reject_negative_audio_length():
    with pytest.raises(ValueError):
        MDXCSeparator._roformer_chunk_starts(audio_length=-1, chunk_size=4, step=2)


def test_short_audio_override_does_not_mutate_separator():
    separator = object.__new__(MDXCSeparator)
    separator.override_model_segment_size = False
    separator.logger = Mock()

    assert separator._use_model_segment_override(5.0) is True
    assert separator.override_model_segment_size is False
    assert separator.logger.warning.call_count == 2
    assert separator._use_model_segment_override(20.0) is False
    assert separator._use_model_segment_override(10.0) is False


def test_explicit_segment_override_applies_to_all_inputs():
    separator = object.__new__(MDXCSeparator)
    separator.override_model_segment_size = True
    separator.logger = Mock()

    assert separator._use_model_segment_override(5.0) is True
    assert separator._use_model_segment_override(20.0) is True
    separator.logger.warning.assert_not_called()
