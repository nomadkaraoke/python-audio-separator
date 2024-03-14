from numpy._typing import NDArray
from audio_separator.separator.common_separator import CommonSeparator

import torch
import os
from ml_collections import ConfigDict

from ..uvr_lib_v5.tfc_tdf_v3 import TFC_TDF_net
from ..uvr_lib_v5 import spec_utils

import librosa
import numpy as np
import audioread

from tqdm import *


class MDXCSeparator(CommonSeparator):
    def __init__(self, common_config, arch_config):
        super().__init__(config=common_config)
        self.segment_size = arch_config.get("segment_size")
        self.overlap = arch_config.get("overlap")
        self.batch_size = arch_config.get("batch_size", 1)

        self.logger.debug(f"MDXC arch params: batch_size={self.batch_size}, segment_size={self.segment_size}, overlap={self.overlap}")

        # Loading the model for inference
        self.logger.debug("Loading Checkpoint model for inference...")
        other_metadata = {"segment_size": self.segment_size, "overlap_mdx23": self.overlap, "batch_size": self.batch_size}
        self.device = self.torch_device
        self.init_metadata()
        self.update_metadata(other_metadata)
        self.load_model(common_config)

    def init_metadata(self):
        prams = {"is_mdx_c_seg_def": False, "segment_size": 256, "batch_size": 1, "overlap_mdx23": 8, "semitone_shift": 0}
        self.other_metadata = prams

    def update_metadata(self, other_metadata):
        self.other_metadata.update(other_metadata)

    def load_model(self, common_config):
        model_path = common_config["model_path"]
        model_data = common_config["model_data"]
        model_data = ConfigDict(model_data)
        device = self.torch_device
        model = TFC_TDF_net(model_data, device=device)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.to(device).eval()
        self.model_data = model_data
        self.model_run = model

    def rerun_mp3(self, audio_file: NDArray, sample_rate: int = 44100):
        with audioread.audio_open(audio_file) as f:
            track_length = int(f.duration)
        return librosa.load(audio_file, duration=track_length, mono=False, sr=sample_rate)[0]

    def demix(self, mix: np.ndarray, prams: dict, model: torch.nn.Module, model_data: ConfigDict, device: str = "cpu") -> dict:

        sr_pitched = 441000
        org_mix = mix
        semitone_shift = prams["semitone_shift"]
        if semitone_shift != 0:
            mix, sr_pitched = spec_utils.change_pitch_semitones(mix, 44100, semitone_shift=-semitone_shift)

        mix = torch.tensor(mix, dtype=torch.float32)

        try:
            S = model.num_target_instruments
        except Exception as e:
            S = model.module.num_target_instruments

        if prams["is_mdx_c_seg_def"]:
            mdx_segment_size = model_data.inference.dim_t
        else:
            mdx_segment_size = prams["segment_size"]

        batch_size = prams["batch_size"]
        chunk_size = model_data.audio.hop_length * (mdx_segment_size - 1)
        overlap = prams["overlap_mdx23"]

        hop_size = chunk_size // overlap
        mix_shape = mix.shape[1]
        pad_size = hop_size - (mix_shape - chunk_size) % hop_size
        mix = torch.cat([torch.zeros(2, chunk_size - hop_size), mix, torch.zeros(2, pad_size + chunk_size - hop_size)], 1)

        chunks = mix.unfold(1, chunk_size, hop_size).transpose(0, 1)
        batches = [chunks[i : i + batch_size] for i in range(0, len(chunks), batch_size)]

        X = torch.zeros(S, *mix.shape) if S > 1 else torch.zeros_like(mix)
        X = X.to(device)

        with torch.no_grad():
            cnt = 0
            for batch in tqdm(batches):
                x = model(batch.to(device))
                for w in x:
                    X[..., cnt * hop_size : cnt * hop_size + chunk_size] += w
                    cnt += 1

        estimated_sources = X[..., chunk_size - hop_size : -(pad_size + chunk_size - hop_size)] / overlap
        del X
        pitch_fix = lambda s: pitch_fix(s, sr_pitched, org_mix, semitone_shift)

        if S > 1:
            sources = {k: pitch_fix(v) if semitone_shift != 0 else v for k, v in zip(model_data.training.instruments, estimated_sources.cpu().detach().numpy())}
            del estimated_sources
            return sources

        est_s = estimated_sources.cpu().detach().numpy()
        del estimated_sources

        if semitone_shift != 0:
            return pitch_fix(est_s)
        else:
            return est_s

    def rename_stems(self, stems: dict) -> dict:
        return {k.lower(): v for k, v in stems.items()}

    def separate(self, audio_file_path):
        self.primary_source = None
        self.secondary_source = None
        self.audio_file_path = audio_file_path
        self.audio_file_base = os.path.splitext(os.path.basename(audio_file_path))[0]

        self.logger.debug("Preparing mix...")
        mix = self.prepare_mix(self.audio_file_path)

        self.logger.debug("Normalizing mix before demixing...")
        mix = spec_utils.normalize(wave=mix, max_peak=self.normalization_threshold)

        source = self.demix(mix, self.other_metadata, self.model_run, self.model_data, self.torch_device)
        stems = self.rename_stems(source)

        output_files = []

        self.logger.debug("Processing output files...")

        if not isinstance(self.primary_source, np.ndarray):
            self.logger.debug("Normalizing primary source...")
            self.primary_source = spec_utils.normalize(wave=stems["vocals"], max_peak=self.normalization_threshold).T

        if not isinstance(self.secondary_source, np.ndarray):
            self.logger.debug("Normalizing primary source...")
            self.secondary_source = spec_utils.normalize(wave=stems["instrumental"], max_peak=self.normalization_threshold).T

        if not self.output_single_stem or self.output_single_stem.lower() == self.secondary_stem_name.lower():
            self.logger.info(f"Saving {self.secondary_stem_name} Instrumental stem...")
            if not self.secondary_stem_output_path:
                self.secondary_stem_output_path = os.path.join(f"{self.audio_file_base}_({self.secondary_stem_name})_{self.model_name}.{self.output_format.lower()}")
            self.secondary_source_map = self.final_process(self.secondary_stem_output_path, self.secondary_source, self.secondary_stem_name)
            output_files.append(self.secondary_stem_output_path)

        if not self.output_single_stem or self.output_single_stem.lower() == self.primary_stem_name.lower():
            self.logger.info(f"Saving {self.primary_stem_name} Vocals stem...")
            if not self.primary_stem_output_path:
                self.primary_stem_output_path = os.path.join(f"{self.audio_file_base}_({self.primary_stem_name})_{self.model_name}.{self.output_format.lower()}")
            if not isinstance(self.primary_source, np.ndarray):
                self.primary_source = source.T
            self.primary_source_map = self.final_process(self.primary_stem_output_path, self.primary_source, self.primary_stem_name)
            output_files.append(self.primary_stem_output_path)
        return output_files
