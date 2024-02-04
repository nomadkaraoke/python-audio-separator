"""Module for separating audio sources using VR architecture models."""

import os
import torch
import librosa
import onnxruntime as ort
import numpy as np
import onnx2torch
from audio_separator.separator import spec_utils
from audio_separator.separator.stft import STFT
from audio_separator.separator.common_separator import CommonSeparator


class VRSeparator(CommonSeparator):
    """
    VRSeparator is responsible for separating audio sources using VR models.
    It initializes with configuration parameters and prepares the model for separation tasks.
    """

    def __init__(self, common_config, arch_config):
        super().__init__(config=common_config)

        self.hop_length = arch_config.get("hop_length")
        self.segment_size = arch_config.get("segment_size")
        self.overlap = arch_config.get("overlap")
        self.batch_size = arch_config.get("batch_size")

        self.logger.debug(f"Model params: primary_stem={self.primary_stem_name}, secondary_stem={self.secondary_stem_name}")
        self.logger.debug(f"Model params: batch_size={self.batch_size}, compensate={self.compensate}, segment_size={self.segment_size}, dim_f={self.dim_f}, dim_t={self.dim_t}")
        self.logger.debug(f"Model params: n_fft={self.n_fft}, hop={self.hop_length}")

        # Loading the model for inference
        self.logger.debug("Loading ONNX model for inference...")
        if self.segment_size == self.dim_t:
            ort_ = ort.InferenceSession(self.model_path, providers=self.onnx_execution_provider)
            self.model_run = lambda spek: ort_.run(None, {"input": spek.cpu().numpy()})[0]
            self.logger.debug("Model loaded successfully using ONNXruntime inferencing session.")
        else:
            self.model_run = onnx2torch.convert(self.model_path)
            self.model_run.to(self.torch_device).eval()
            self.logger.warning("Model converted from onnx to pytorch due to segment size not matching dim_t, processing may be slower.")

        self.n_bins = None
        self.trim = None
        self.chunk_size = None
        self.gen_size = None
        self.stft = None

        self.primary_source = None
        self.secondary_source = None
        self.audio_file_path = None
        self.audio_file_base = None
        self.secondary_source_map = None
        self.primary_source_map = None
