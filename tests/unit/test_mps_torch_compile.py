from unittest.mock import Mock, patch

import pytest
import torch

from audio_separator.separator.architectures.mdxc_separator import MDXCSeparator


def _separator(*, use_torch_compile=True, policy_allows_compile=True):
    separator = object.__new__(MDXCSeparator)
    separator.logger = Mock()
    separator.use_torch_compile = use_torch_compile
    separator._should_torch_compile = use_torch_compile and policy_allows_compile
    separator.effective_torch_compile = False
    separator.torch_device = torch.device("mps")
    separator.model_run = Mock()
    separator.model_run.layers = torch.nn.ModuleList([torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])])
    return separator


def test_regional_compile_wraps_repeated_transformers():
    separator = _separator()

    with patch.object(torch.nn.Module, "compile", autospec=True) as compile_module:
        separator._configure_model_compilation()

    assert separator.is_torch_compiled is True
    assert compile_module.call_count == 2
    separator.logger.warning.assert_not_called()


def test_regional_compile_skips_when_policy_does_not_enable_it():
    separator = _separator(policy_allows_compile=False)

    with patch.object(torch.nn.Module, "compile", autospec=True) as compile_module:
        separator._configure_model_compilation()

    assert separator.is_torch_compiled is False
    assert separator.effective_torch_compile is False
    compile_module.assert_not_called()
    separator.logger.warning.assert_not_called()


def test_disabled_regional_compile_is_silent():
    separator = _separator(use_torch_compile=False)

    with patch.object(torch.nn.Module, "compile", autospec=True) as compile_module:
        separator._configure_model_compilation()

    assert separator.is_torch_compiled is False
    compile_module.assert_not_called()
    separator.logger.warning.assert_not_called()


def test_regional_compile_requires_restorable_module_calls():
    separator = _separator()
    transformer = Mock(spec=["compile"])
    separator.model_run.layers = [[transformer]]

    separator._configure_model_compilation()

    assert separator.is_torch_compiled is False
    transformer.compile.assert_not_called()
    separator.logger.warning.assert_called_once()


def test_regional_compile_falls_back_to_eager_when_compilation_fails():
    separator = _separator()
    transformer_blocks = [transformer for layer in separator.model_run.layers for transformer in layer]
    existing_call = Mock()
    transformer_blocks[0]._compiled_call_impl = existing_call
    compile_calls = 0

    def compile_then_fail(module):
        nonlocal compile_calls
        compile_calls += 1
        if compile_calls == 1:
            module._compiled_call_impl = Mock()
            return
        raise RuntimeError("unsupported")

    with patch.object(torch.nn.Module, "compile", autospec=True, side_effect=compile_then_fail):
        separator._configure_model_compilation()

    assert separator.is_torch_compiled is False
    assert transformer_blocks[0]._compiled_call_impl is existing_call
    assert transformer_blocks[1]._compiled_call_impl is None
    separator.logger.warning.assert_called_once()


def test_regional_compile_restores_original_calls_after_lazy_failure():
    separator = _separator()
    transformer_blocks = separator._regional_compile_targets()
    original_calls = [Mock(), Mock()]
    for transformer, original_call in zip(transformer_blocks, original_calls, strict=True):
        transformer._compiled_call_impl = original_call

    def install_compiled_call(module):
        module._compiled_call_impl = Mock()

    with patch.object(torch.nn.Module, "compile", autospec=True, side_effect=install_compiled_call):
        separator._configure_model_compilation()

    assert separator.is_torch_compiled is True
    assert all(
        transformer._compiled_call_impl is not original_call
        for transformer, original_call in zip(transformer_blocks, original_calls, strict=True)
    )

    expected = torch.ones(1, 2, 8)
    separator.model_run.side_effect = [RuntimeError("backend compilation failed"), (expected,)]

    assert separator._run_roformer_model(torch.zeros(2, 8)) is expected
    assert separator.is_torch_compiled is False
    assert all(
        transformer._compiled_call_impl is original_call
        for transformer, original_call in zip(transformer_blocks, original_calls, strict=True)
    )
    separator.logger.warning.assert_called_once()


def test_regional_compile_retries_eager_when_lazy_compilation_fails():
    separator = _separator()
    transformer_blocks = separator._regional_compile_targets()
    for transformer in transformer_blocks:
        transformer._compiled_call_impl = Mock()
    separator.is_torch_compiled = True
    expected = torch.ones(1, 2, 8)
    separator.model_run.side_effect = [RuntimeError("backend compilation failed"), (expected,)]

    result = separator._run_roformer_model(torch.zeros(2, 8))

    assert result is expected
    assert separator.model_run.call_count == 2
    assert separator.is_torch_compiled is False
    assert all(transformer._compiled_call_impl is None for transformer in transformer_blocks)
    separator.logger.warning.assert_called_once()


def test_regional_compile_can_fall_back_after_an_earlier_successful_forward():
    separator = _separator()
    transformer_blocks = separator._regional_compile_targets()
    for transformer in transformer_blocks:
        transformer._compiled_call_impl = Mock()
    separator.is_torch_compiled = True
    first = torch.ones(1, 2, 8)
    fallback = torch.full((1, 2, 8), 2.0)
    separator.model_run.side_effect = [(first,), RuntimeError("recompile failed"), (fallback,)]

    assert separator._run_roformer_model(torch.zeros(2, 8)) is first
    assert separator._run_roformer_model(torch.zeros(2, 8)) is fallback

    assert separator.model_run.call_count == 3
    assert separator.is_torch_compiled is False
    assert all(transformer._compiled_call_impl is None for transformer in transformer_blocks)
    separator.logger.warning.assert_called_once()


def test_regional_compile_preserves_an_eager_retry_error():
    separator = _separator()
    for transformer in separator._regional_compile_targets():
        transformer._compiled_call_impl = Mock()
    separator.is_torch_compiled = True
    separator.model_run.side_effect = [RuntimeError("compiled path failed"), ValueError("model failed")]

    with pytest.raises(ValueError, match="model failed"):
        separator._run_roformer_model(torch.zeros(2, 8))

    assert separator.model_run.call_count == 2
    assert separator.is_torch_compiled is False
    separator.logger.warning.assert_not_called()


def test_eager_roformer_errors_are_not_retried():
    separator = _separator()
    separator.is_torch_compiled = False
    separator.model_run.side_effect = RuntimeError("model failed")

    with pytest.raises(RuntimeError, match="model failed"):
        separator._run_roformer_model(torch.zeros(2, 8))

    separator.model_run.assert_called_once()
    separator.logger.warning.assert_not_called()
