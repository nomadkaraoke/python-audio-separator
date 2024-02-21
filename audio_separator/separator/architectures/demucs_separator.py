import os
import torch
import numpy as np
from pathlib import Path
from audio_separator.separator.common_separator import CommonSeparator
from audio_separator.separator.uvr_lib_v5.demucs.apply import apply_model, demucs_segments
from audio_separator.separator.uvr_lib_v5.demucs.hdemucs import HDemucs
from audio_separator.separator.uvr_lib_v5.demucs.pretrained import get_model as get_demucs_model
from audio_separator.separator.uvr_lib_v5 import spec_utils

DEMUCS_2_SOURCE = ["instrumental", "vocals"]
DEMUCS_4_SOURCE = ["drums", "bass", "other", "vocals"]

DEMUCS_2_SOURCE_MAPPER = {CommonSeparator.INST_STEM: 0, CommonSeparator.VOCAL_STEM: 1}

DEMUCS_4_SOURCE_MAPPER = {CommonSeparator.BASS_STEM: 0, CommonSeparator.DRUM_STEM: 1, CommonSeparator.OTHER_STEM: 2, CommonSeparator.VOCAL_STEM: 3}

DEMUCS_6_SOURCE_MAPPER = {
    CommonSeparator.BASS_STEM: 0,
    CommonSeparator.DRUM_STEM: 1,
    CommonSeparator.OTHER_STEM: 2,
    CommonSeparator.VOCAL_STEM: 3,
    CommonSeparator.GUITAR_STEM: 4,
    CommonSeparator.PIANO_STEM: 5,
}

DEMUCS_4_SOURCE_LIST = [CommonSeparator.BASS_STEM, CommonSeparator.DRUM_STEM, CommonSeparator.OTHER_STEM, CommonSeparator.VOCAL_STEM]
DEMUCS_6_SOURCE_LIST = [CommonSeparator.BASS_STEM, CommonSeparator.DRUM_STEM, CommonSeparator.OTHER_STEM, CommonSeparator.VOCAL_STEM, CommonSeparator.GUITAR_STEM, CommonSeparator.PIANO_STEM]


class DemucsSeparator(CommonSeparator):
    def __init__(self, common_config, arch_config):
        # Any configuration values which can be shared between architectures should be set already in CommonSeparator,
        # e.g. user-specified functionality choices (self.output_single_stem) or common model parameters (self.primary_stem_name)
        super().__init__(config=common_config)

        # Initializing user-configurable parameters, passed through with an mdx_from the CLI or Separator instance

        # 'Select a stem for extraction with the chosen model:\n\n'
        # '• All Stems - Extracts all available stems.\n'
        # '• Vocals - Only the "vocals" stem.\n'
        # '• Other - Only the "other" stem.\n'
        # '• Bass - Only the "bass" stem.\n'
        # '• Drums - Only the "drums" stem.'
        self.selected_stem = arch_config.get("selected_stem", [CommonSeparator.ALL_STEMS])

        # Adjust segments to manage RAM or V-RAM usage:
        # - Smaller sizes consume less resources.
        # - Bigger sizes consume more resources, but may provide better results.
        # - "Default" picks the optimal size.
        # DEMUCS_SEGMENTS = (DEF_OPT, '1', '5', '10', '15', '20',
        #           '25', '30', '35', '40', '45', '50',
        #           '55', '60', '65', '70', '75', '80',
        #           '85', '90', '95', '100')
        self.segment_size = arch_config.get("segment_size", "Default")

        # Performs multiple predictions with random shifts of the input and averages them.
        # The higher number of shifts, the longer the prediction will take.
        # Not recommended unless you have a GPU.
        # DEMUCS_SHIFTS = (0, 1, 2, 3, 4, 5,
        #                 6, 7, 8, 9, 10, 11,
        #                 12, 13, 14, 15, 16, 17,
        #                 18, 19, 20)
        self.shifts = arch_config.get("shifts", 2)

        # This option controls the amount of overlap between prediction windows.
        #  - Higher values can provide better results, but will lead to longer processing times.
        #  - You can choose between 0.001-0.999
        # DEMUCS_OVERLAP = (0.25, 0.50, 0.75, 0.99)
        self.overlap = arch_config.get("overlap", 0.25)

        # Enables "Segments". Deselecting this option is only recommended for those with powerful PCs.
        self.segments_enabled = arch_config.get("segments_enabled", 2)

        self.demucs_source_list, self.demucs_source_map, self.demucs_stem_count = DEMUCS_4_SOURCE, DEMUCS_4_SOURCE_MAPPER, 4

        self.primary_stem = CommonSeparator.PRIMARY_STEM if self.selected_stem == CommonSeparator.ALL_STEMS else self.selected_stem
        self.secondary_stem = CommonSeparator.SECONDARY_STEM

        self.audio_file_path = None
        self.audio_file_base = None
        self.demucs_model_instance = None

    def separate(self, audio_file_path):
        self.logger.debug("SeperateDemucs: Starting separation process...")
        source = None
        stem_source = None
        inst_source = {}

        self.audio_file_path = audio_file_path
        self.audio_file_base = os.path.splitext(os.path.basename(audio_file_path))[0]

        # Prepare the mix for processing
        self.logger.debug("Preparing mix...")
        mix = self.prepare_mix(self.audio_file_path)

        self.logger.debug(f"SeperateDemucs: Mix prepared for demixing. Shape: {mix.shape}")

        self.logger.debug("SeperateDemucs: Loading model for demixing...")

        self.demucs_model_instance = HDemucs(sources=self.demucs_source_list)
        self.demucs_model_instance = get_demucs_model(name=os.path.splitext(os.path.basename(self.model_path))[0], repo=Path(os.path.dirname(self.model_path)))
        self.demucs_model_instance = demucs_segments(self.segment_size, self.demucs_model_instance)
        self.demucs_model_instance.to(self.torch_device)
        self.demucs_model_instance.eval()

        self.logger.debug("SeperateDemucs: Model loaded and set to evaluation mode.")

        source = self.demix_demucs(mix)

        del self.demucs_model_instance
        self.clear_gpu_cache()
        self.logger.debug("SeperateDemucs: Model and GPU cache cleared after demixing.")

        if isinstance(inst_source, np.ndarray):
            self.logger.debug("SeperateDemucs: Processing instance source...")
            source_reshape = spec_utils.reshape_sources(inst_source[self.demucs_source_map[CommonSeparator.VOCAL_STEM]], source[self.demucs_source_map[CommonSeparator.VOCAL_STEM]])
            inst_source[self.demucs_source_map[CommonSeparator.VOCAL_STEM]] = source_reshape
            source = inst_source

        if isinstance(source, np.ndarray):
            self.logger.debug(f"SeperateDemucs: Processing source array, source length is {len(source)}")
            if len(source) == 2:
                self.logger.debug("SeperateDemucs: Setting source map to 2-stem...")
                self.demucs_source_map = DEMUCS_2_SOURCE_MAPPER
            else:
                self.logger.debug("SeperateDemucs: Setting source map to 4 or 6-stem...")
                self.demucs_source_map = DEMUCS_6_SOURCE_MAPPER if len(source) == 6 else DEMUCS_4_SOURCE_MAPPER

        # if self.selected_stem == CommonSeparator.ALL_STEMS:

        self.logger.debug("SeperateDemucs: Processing for all stems...")
        for stem_name, stem_value in self.demucs_source_map.items():
            stem_path = os.path.join(f"{self.audio_file_base}_({stem_name})_{self.model_name}.{self.output_format.lower()}")
            stem_source = source[stem_value].T

            self.final_process(stem_path, stem_source, stem_name)

        # else:
        #     def secondary_save(sec_stem_name, source, raw_mixture=None, is_inst_mixture=False):
        #         self.logger.debug(f"SeperateDemucs: Saving secondary stem: {sec_stem_name}")
        #         secondary_source = self.secondary_source if not is_inst_mixture else None
        #         secondary_stem_path = os.path.join(self.export_path, f"{self.audio_file_base}_({sec_stem_name}).wav")
        #         secondary_source_secondary = None

        #         if not isinstance(secondary_source, np.ndarray):
        #             if self.is_demucs_combine_stems:
        #                 source = list(source)
        #                 if is_inst_mixture:
        #                     source = [i for n, i in enumerate(source) if not n in [self.demucs_source_map[self.primary_stem], self.demucs_source_map[CommonSeparator.VOCAL_STEM]]]
        #                 else:
        #                     source.pop(self.demucs_source_map[self.primary_stem])

        #                 source = source[: len(source) - 2] if is_no_piano_guitar else source
        #                 secondary_source = np.zeros_like(source[0])
        #                 for i in source:
        #                     secondary_source += i
        #                 secondary_source = secondary_source.T
        #             else:
        #                 if not isinstance(raw_mixture, np.ndarray):
        #                     raw_mixture = self.prepare_mix(self.audio_file)

        #                 secondary_source = source[self.demucs_source_map[self.primary_stem]]

        #                 if self.is_invert_spec:
        #                     secondary_source = spec_utils.invert_stem(raw_mixture, secondary_source)
        #                 else:
        #                     raw_mixture = spec_utils.reshape_sources(secondary_source, raw_mixture)
        #                     secondary_source = -secondary_source.T + raw_mixture.T

        #         if not is_inst_mixture:
        #             self.secondary_source = secondary_source
        #             secondary_source_secondary = self.secondary_source_secondary
        #             self.secondary_source = self.process_secondary_stem(secondary_source, secondary_source_secondary)
        #             self.secondary_source_map = {self.secondary_stem: self.secondary_source}

        #         self.write_audio(secondary_stem_path, secondary_source, samplerate, stem_name=sec_stem_name)

        #     secondary_save(self.secondary_stem, source, raw_mixture=mix)

        #     if self.is_demucs_pre_proc_model_inst_mix and self.pre_proc_model and not self.is_4_stem_ensemble:
        #         secondary_save(f"{self.secondary_stem} {CommonSeparator.INST_STEM}", source, raw_mixture=inst_mix, is_inst_mixture=True)

        #     if not self.is_secondary_stem_only:
        #         primary_stem_path = os.path.join(self.export_path, f"{self.audio_file_base}_({self.primary_stem}).wav")
        #         if not isinstance(self.primary_source, np.ndarray):
        #             self.primary_source = source[self.demucs_source_map[self.primary_stem]].T

        #         self.primary_source_map = self.final_process(primary_stem_path, self.primary_source, self.secondary_source_primary, self.primary_stem, samplerate)

        #     secondary_sources = {**self.primary_source_map, **self.secondary_source_map}

        #     self.process_vocal_split_chain(secondary_sources)

        #     if self.is_secondary_model:
        #         return secondary_sources

    def demix_demucs(self, mix):
        """
        Demixes the input mix using the demucs model.
        """
        self.logger.debug("SeperateDemucs: Starting demixing process in demix_demucs...")
        org_mix = mix

        # if self.is_pitch_change:
        #     self.logger.debug("SeperateDemucs: Applying pitch change...")
        #     mix, sr_pitched = spec_utils.change_pitch_semitones(mix, 44100, semitone_shift=-self.semitone_shift)

        processed = {}
        mix = torch.tensor(mix, dtype=torch.float32)
        ref = mix.mean(0)
        mix = (mix - ref.mean()) / ref.std()
        mix_infer = mix

        with torch.no_grad():
            self.logger.debug("SeperateDemucs: Running model inference...")
            sources = apply_model(
                model=self.demucs_model_instance,
                mix=mix_infer[None],
                shifts=self.shifts,
                split=self.segments_enabled,
                overlap=self.overlap,
                static_shifts=1 if self.shifts == 0 else self.shifts,
                set_progress_bar=None,
                device=self.torch_device,
            )[0]

        sources = (sources * ref.std() + ref.mean()).cpu().numpy()
        sources[[0, 1]] = sources[[1, 0]]
        processed[mix] = sources[:, :, 0:None].copy()
        sources = list(processed.values())
        sources = [s[:, :, 0:None] for s in sources]
        # sources = [self.pitch_fix(s[:,:,0:None], sr_pitched, org_mix) if self.is_pitch_change else s[:,:,0:None] for s in sources]
        sources = np.concatenate(sources, axis=-1)

        # if self.is_pitch_change:
        #     self.logger.debug("SeperateDemucs: Fixing pitch post-demixing...")
        #     sources = np.stack([self.pitch_fix(stem, sr_pitched, org_mix) for stem in sources])

        return sources
