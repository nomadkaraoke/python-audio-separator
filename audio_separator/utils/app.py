import os
import torch
import shutil
import logging
import gradio as gr

from audio_separator.separator import Separator

device = "cuda" if torch.cuda.is_available() else "cpu"
use_autocast = device == "cuda"

#=========================#
#     Roformer Models     #
#=========================#
ROFORMER_MODELS = {
    'BS-Roformer-Viperx-1053': 'model_bs_roformer_ep_937_sdr_10.5309.ckpt',
    'BS-Roformer-Viperx-1296': 'model_bs_roformer_ep_368_sdr_12.9628.ckpt',
    'BS-Roformer-Viperx-1297': 'model_bs_roformer_ep_317_sdr_12.9755.ckpt',
    'BS-Roformer-De-Reverb': 'deverb_bs_roformer_8_384dim_10depth.ckpt',
    'Mel-Roformer-Viperx-1143': 'model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt',
    'Mel-Roformer-Crowd-Aufr33-Viperx': 'mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt',
    'Mel-Roformer-Karaoke-Aufr33-Viperx': 'mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt',
    'Mel-Roformer-Denoise-Aufr33': 'denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt',
    'Mel-Roformer-Denoise-Aufr33-Aggr': 'denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt',
    'MelBand Roformer Kim | Inst V1 by Unwa': 'melband_roformer_inst_v1.ckpt',
    'MelBand Roformer Kim | Inst V2 by Unwa': 'melband_roformer_inst_v2.ckpt',
    'MelBand Roformer Kim | InstVoc Duality V1 by Unwa': 'melband_roformer_instvoc_duality_v1.ckpt',
    'MelBand Roformer Kim | InstVoc Duality V2 by Unwa': 'melband_roformer_instvox_duality_v2.ckpt',
}
#=========================#
#      MDX23C Models      #
#=========================#
MDX23C_MODELS = [
    'MDX23C-8KFFT-InstVoc_HQ.ckpt',
    'MDX23C-8KFFT-InstVoc_HQ_2.ckpt',
    'MDX23C_D1581.ckpt',
]
#=========================#
#     MDXN-NET Models     #
#=========================#
MDXNET_MODELS = [
    'UVR-MDX-NET-Inst_1.onnx',
    'UVR-MDX-NET-Inst_2.onnx',
    'UVR-MDX-NET-Inst_3.onnx',
    'UVR-MDX-NET-Inst_HQ_1.onnx',
    'UVR-MDX-NET-Inst_HQ_2.onnx',
    'UVR-MDX-NET-Inst_HQ_3.onnx',
    'UVR-MDX-NET-Inst_HQ_4.onnx',
    'UVR-MDX-NET-Inst_HQ_5.onnx',
    'UVR-MDX-NET_Inst_82_beta.onnx',
    'UVR-MDX-NET_Inst_90_beta.onnx',
    'UVR-MDX-NET_Inst_187_beta.onnx',
    'UVR-MDX-NET-Inst_full_292.onnx',
    'UVR-MDX-NET_Main_340.onnx',
    'UVR-MDX-NET_Main_390.onnx',
    'UVR-MDX-NET_Main_406.onnx',
    'UVR-MDX-NET_Main_427.onnx',
    'UVR-MDX-NET_Main_438.onnx',
    'UVR-MDX-NET-Crowd_HQ_1.onnx',
    'UVR-MDX-NET-Voc_FT.onnx',
    'UVR_MDXNET_1_9703.onnx',
    'UVR_MDXNET_2_9682.onnx',
    'UVR_MDXNET_3_9662.onnx',
    'UVR_MDXNET_9482.onnx',
    'UVR_MDXNET_KARA.onnx',
    'UVR_MDXNET_KARA_2.onnx',
    'UVR_MDXNET_Main.onnx',
    'kuielab_a_bass.onnx',
    'kuielab_a_drums.onnx',
    'kuielab_a_other.onnx',
    'kuielab_a_vocals.onnx',
    'kuielab_b_bass.onnx',
    'kuielab_b_drums.onnx',
    'kuielab_b_other.onnx',
    'kuielab_b_vocals.onnx',
    'Kim_Inst.onnx',
    'Kim_Vocal_1.onnx',
    'Kim_Vocal_2.onnx',
    'Reverb_HQ_By_FoxJoy.onnx',
]
#========================#
#     VR-ARCH Models     #
#========================#
VR_ARCH_MODELS = [
    '1_HP-UVR.pth',
    '2_HP-UVR.pth',
    '3_HP-Vocal-UVR.pth',
    '4_HP-Vocal-UVR.pth',
    '5_HP-Karaoke-UVR.pth',
    '6_HP-Karaoke-UVR.pth',
    '7_HP2-UVR.pth',
    '8_HP2-UVR.pth',
    '9_HP2-UVR.pth',
    '10_SP-UVR-2B-32000-1.pth',
    '11_SP-UVR-2B-32000-2.pth',
    '12_SP-UVR-3B-44100.pth',
    '13_SP-UVR-4B-44100-1.pth',
    '14_SP-UVR-4B-44100-2.pth',
    '15_SP-UVR-MID-44100-1.pth',
    '16_SP-UVR-MID-44100-2.pth',
    '17_HP-Wind_Inst-UVR.pth',
    'MGM_HIGHEND_v4.pth',
    'MGM_LOWEND_A_v4.pth',
    'MGM_LOWEND_B_v4.pth',
    'MGM_MAIN_v4.pth',
    'UVR-BVE-4B_SN-44100-1.pth',
    'UVR-DeEcho-DeReverb.pth',
    'UVR-De-Echo-Aggressive.pth',
    'UVR-De-Echo-Normal.pth',
    'UVR-DeNoise-Lite.pth',
    'UVR-DeNoise.pth',
]
#=======================#
#     DEMUCS Models     #
#=======================#
DEMUCS_MODELS = [
    'hdemucs_mmi.yaml',
    'htdemucs.yaml',
    'htdemucs_6s.yaml',
    'htdemucs_ft.yaml',
]

def print_message(input_file, model_name):
    """Prints information about the audio separation process."""
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    print("\n")
    print("🎵 Audio-Separator 🎵")
    print("Input audio:", base_name)
    print("Separation Model:", model_name)
    print("Audio Separation Process...")

def prepare_output_dir(input_file, output_dir):
    """Create a directory for the output files and clean it if it already exists."""
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    out_dir = os.path.join(output_dir, base_name)
    try:
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir)
    except Exception as e:
        raise RuntimeError(f"Failed to prepare output directory {out_dir}: {e}")
    return out_dir

def rename_stems(audio, vocals_stem, instrumental_stem, other_stem, drums_stem, bass_stem, guitar_stem, piano_stem, model):
    base_name = os.path.splitext(os.path.basename(audio))[0]
    stems = {
        "Vocals": vocals_stem.replace("NAME", base_name).replace("STEM", "Vocals").replace("MODEL", model),
        "Instrumental": instrumental_stem.replace("NAME", base_name).replace("STEM", "Instrumental").replace("MODEL", model),
        "Drums": drums_stem.replace("NAME", base_name).replace("STEM", "Drums").replace("MODEL", model),
        "Bass": bass_stem.replace("NAME", base_name).replace("STEM", "Bass").replace("MODEL", model),
        "Other": other_stem.replace("NAME", base_name).replace("STEM", "Other").replace("MODEL", model),
        "Guitar": guitar_stem.replace("NAME", base_name).replace("STEM", "Guitar").replace("MODEL", model),
        "Piano": piano_stem.replace("NAME", base_name).replace("STEM", "Piano").replace("MODEL", model),
    }
    return stems

def roformer_separator(audio, model_key, seg_size, override_seg_size, overlap, pitch_shift, model_dir, out_dir, out_format, norm_thresh, amp_thresh, batch_size, vocals_stem, instrumental_stem, other_stem, drums_stem, bass_stem, guitar_stem, piano_stem, progress=gr.Progress(track_tqdm=True)):
    """Separate audio using Roformer model."""
    stemname = rename_stems(audio, vocals_stem, instrumental_stem, other_stem, drums_stem, bass_stem, guitar_stem, piano_stem, model_key)
    print_message(audio, model_key)
    model = ROFORMER_MODELS[model_key]
    try:
        out_dir = prepare_output_dir(audio, out_dir)
        separator = Separator(
            log_level=logging.WARNING,
            model_file_dir=model_dir,
            output_dir=out_dir,
            output_format=out_format,
            normalization_threshold=norm_thresh,
            amplification_threshold=amp_thresh,
            use_autocast=use_autocast,
            mdxc_params={
                "segment_size": seg_size,
                "override_model_segment_size": override_seg_size,
                "batch_size": batch_size,
                "overlap": overlap,
                "pitch_shift": pitch_shift,
            }
        )

        progress(0.2, desc="Model loaded...")
        separator.load_model(model_filename=model)

        progress(0.7, desc="Audio separated...")
        separation = separator.separate(audio, stemname)
        print(f"Separation complete!\nResults: {', '.join(separation)}")

        stems = [os.path.join(out_dir, file_name) for file_name in separation]
        return stems[0], stems[1]
    except Exception as e:
        raise RuntimeError(f"Roformer separation failed: {e}") from e

def mdx23c_separator(audio, model, seg_size, override_seg_size, overlap, pitch_shift, model_dir, out_dir, out_format, norm_thresh, amp_thresh, batch_size, vocals_stem, instrumental_stem, other_stem, drums_stem, bass_stem, guitar_stem, piano_stem, progress=gr.Progress(track_tqdm=True)):
    """Separate audio using MDX23C model."""
    stemname = rename_stems(audio, vocals_stem, instrumental_stem, other_stem, drums_stem, bass_stem, guitar_stem, piano_stem, model)
    print_message(audio, model)
    try:
        out_dir = prepare_output_dir(audio, out_dir)
        separator = Separator(
            log_level=logging.WARNING,
            model_file_dir=model_dir,
            output_dir=out_dir,
            output_format=out_format,
            normalization_threshold=norm_thresh,
            amplification_threshold=amp_thresh,
            use_autocast=use_autocast,
            mdxc_params={
                "segment_size": seg_size,
                "override_model_segment_size": override_seg_size,
                "batch_size": batch_size,
                "overlap": overlap,
                "pitch_shift": pitch_shift,
            }
        )

        progress(0.2, desc="Model loaded...")
        separator.load_model(model_filename=model)

        progress(0.7, desc="Audio separated...")
        separation = separator.separate(audio, stemname)
        print(f"Separation complete!\nResults: {', '.join(separation)}")

        stems = [os.path.join(out_dir, file_name) for file_name in separation]
        return stems[0], stems[1]
    except Exception as e:
        raise RuntimeError(f"MDX23C separation failed: {e}") from e

def mdx_separator(audio, model, hop_length, seg_size, overlap, denoise, model_dir, out_dir, out_format, norm_thresh, amp_thresh, batch_size, vocals_stem, instrumental_stem, other_stem, drums_stem, bass_stem, guitar_stem, piano_stem, progress=gr.Progress(track_tqdm=True)):
    """Separate audio using MDX-NET model."""
    stemname = rename_stems(audio, vocals_stem, instrumental_stem, other_stem, drums_stem, bass_stem, guitar_stem, piano_stem, model)
    print_message(audio, model)
    try:
        out_dir = prepare_output_dir(audio, out_dir)
        separator = Separator(
            log_level=logging.WARNING,
            model_file_dir=model_dir,
            output_dir=out_dir,
            output_format=out_format,
            normalization_threshold=norm_thresh,
            amplification_threshold=amp_thresh,
            use_autocast=use_autocast,
            mdx_params={
                "hop_length": hop_length,
                "segment_size": seg_size,
                "overlap": overlap,
                "batch_size": batch_size,
                "enable_denoise": denoise,
            }
        )

        progress(0.2, desc="Model loaded...")
        separator.load_model(model_filename=model)

        progress(0.7, desc="Audio separated...")
        separation = separator.separate(audio, stemname)
        print(f"Separation complete!\nResults: {', '.join(separation)}")

        stems = [os.path.join(out_dir, file_name) for file_name in separation]
        return stems[0], stems[1]
    except Exception as e:
        raise RuntimeError(f"MDX-NET separation failed: {e}") from e

def vr_separator(audio, model, window_size, aggression, tta, post_process, post_process_threshold, high_end_process, model_dir, out_dir, out_format, norm_thresh, amp_thresh, batch_size, vocals_stem, instrumental_stem, other_stem, drums_stem, bass_stem, guitar_stem, piano_stem, progress=gr.Progress(track_tqdm=True)):
    """Separate audio using VR ARCH model."""
    stemname = rename_stems(audio, vocals_stem, instrumental_stem, other_stem, drums_stem, bass_stem, guitar_stem, piano_stem, model)
    print_message(audio, model)
    try:
        out_dir = prepare_output_dir(audio, out_dir)
        separator = Separator(
            log_level=logging.WARNING,
            model_file_dir=model_dir,
            output_dir=out_dir,
            output_format=out_format,
            normalization_threshold=norm_thresh,
            amplification_threshold=amp_thresh,
            use_autocast=use_autocast,
            vr_params={
                "batch_size": batch_size,
                "window_size": window_size,
                "aggression": aggression,
                "enable_tta": tta,
                "enable_post_process": post_process,
                "post_process_threshold": post_process_threshold,
                "high_end_process": high_end_process,
            }
        )

        progress(0.2, desc="Model loaded...")
        separator.load_model(model_filename=model)

        progress(0.7, desc="Audio separated...")
        separation = separator.separate(audio, stemname)
        print(f"Separation complete!\nResults: {', '.join(separation)}")

        stems = [os.path.join(out_dir, file_name) for file_name in separation]
        return stems[0], stems[1]
    except Exception as e:
        raise RuntimeError(f"VR ARCH separation failed: {e}") from e

def demucs_separator(audio, model, seg_size, shifts, overlap, segments_enabled, model_dir, out_dir, out_format, norm_thresh, amp_thresh, vocals_stem, instrumental_stem, other_stem, drums_stem, bass_stem, guitar_stem, piano_stem, progress=gr.Progress(track_tqdm=True)):
    """Separate audio using Demucs model."""
    stemname = rename_stems(audio, vocals_stem, instrumental_stem, other_stem, drums_stem, bass_stem, guitar_stem, piano_stem, model)
    print_message(audio, model)
    try:
        out_dir = prepare_output_dir(audio, out_dir)
        separator = Separator(
            log_level=logging.WARNING,
            model_file_dir=model_dir,
            output_dir=out_dir,
            output_format=out_format,
            normalization_threshold=norm_thresh,
            amplification_threshold=amp_thresh,
            use_autocast=use_autocast,
            demucs_params={
                "segment_size": seg_size,
                "shifts": shifts,
                "overlap": overlap,
                "segments_enabled": segments_enabled,
            }
        )

        progress(0.2, desc="Model loaded...")
        separator.load_model(model_filename=model)

        progress(0.7, desc="Audio separated...")
        separation = separator.separate(audio, stemname)
        print(f"Separation complete!\nResults: {', '.join(separation)}")

        stems = [os.path.join(out_dir, file_name) for file_name in separation]

        if model == "htdemucs_6s.yaml":
            return stems[0], stems[1], stems[2], stems[3], stems[4], stems[5]
        else:
            return stems[0], stems[1], stems[2], stems[3], None, None
    except Exception as e:
        raise RuntimeError(f"Demucs separation failed: {e}") from e

def update_stems(model):
    """Update the visibility of stem outputs based on the selected Demucs model."""
    if model == "htdemucs_6s.yaml":
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)

with gr.Blocks(
    title="🎵 Audio-Separator 🎵",
    css="footer{display:none !important}",
    theme=gr.themes.Default(
        spacing_size="sm",
        radius_size="lg",
    )
) as app:
    gr.HTML("<h1> 🎵 Audio-Separator 🎵 </h1>")

    with gr.Tab("Roformer"):
        with gr.Group():
            with gr.Row():
                roformer_model = gr.Dropdown(label="Select the Model", choices=list(ROFORMER_MODELS.keys()))
            with gr.Row():
                roformer_seg_size = gr.Slider(minimum=32, maximum=4000, step=32, value=256, label="Segment Size", info="Larger consumes more resources, but may give better results.")
                roformer_override_seg_size = gr.Checkbox(value=False, label="Override segment size", info="Override model default segment size instead of using the model default value.")
                roformer_overlap = gr.Slider(minimum=2, maximum=10, step=1, value=8, label="Overlap", info="Amount of overlap between prediction windows. Lower is better but slower.")
                roformer_pitch_shift = gr.Slider(minimum=-24, maximum=24, step=1, value=0, label="Pitch shift", info="Shift audio pitch by a number of semitones while processing. may improve output for deep/high vocals.")
        with gr.Row():
            roformer_audio = gr.Audio(label="Input Audio", type="filepath")
        with gr.Row():
            roformer_button = gr.Button("Separate!", variant="primary")
        with gr.Row():
            roformer_stem1 = gr.Audio(label="Stem 1", type="filepath", interactive=False)
            roformer_stem2 = gr.Audio(label="Stem 2", type="filepath", interactive=False)

    with gr.Tab("MDX23C"):
        with gr.Group():
            with gr.Row():
                mdx23c_model = gr.Dropdown(label="Select the Model", choices=MDX23C_MODELS)
            with gr.Row():
                mdx23c_seg_size = gr.Slider(minimum=32, maximum=4000, step=32, value=256, label="Segment Size", info="Larger consumes more resources, but may give better results.")
                mdx23c_override_seg_size = gr.Checkbox(value=False, label="Override segment size", info="Override model default segment size instead of using the model default value.")
                mdx23c_overlap = gr.Slider(minimum=2, maximum=50, step=1, value=8, label="Overlap", info="Amount of overlap between prediction windows. Higher is better but slower.")
                mdx23c_pitch_shift = gr.Slider(minimum=-24, maximum=24, step=1, value=0, label="Pitch shift", info="Shift audio pitch by a number of semitones while processing. may improve output for deep/high vocals.")
        with gr.Row():
            mdx23c_audio = gr.Audio(label="Input Audio", type="filepath")
        with gr.Row():
            mdx23c_button = gr.Button("Separate!", variant="primary")
        with gr.Row():
            mdx23c_stem1 = gr.Audio(label="Stem 1", type="filepath", interactive=False)
            mdx23c_stem2 = gr.Audio(label="Stem 2", type="filepath", interactive=False)

    with gr.Tab("MDX-NET"):
        with gr.Group():
            with gr.Row():
                mdx_model = gr.Dropdown(label="Select the Model", choices=MDXNET_MODELS)
            with gr.Row():
                mdx_hop_length = gr.Slider(minimum=32, maximum=2048, step=32, value=1024, label="Hop Length", info="Usually called stride in neural networks; only change if you know what you're doing.")
                mdx_seg_size = gr.Slider(minimum=32, maximum=4000, step=32, value=256, label="Segment Size", info="Larger consumes more resources, but may give better results.")
                mdx_overlap = gr.Slider(minimum=0.001, maximum=0.999, step=0.001, value=0.25, label="Overlap", info="Amount of overlap between prediction windows. Higher is better but slower.")
                mdx_denoise = gr.Checkbox(value=False, label="Denoise", info="Enable denoising after separation.")
        with gr.Row():
            mdx_audio = gr.Audio(label="Input Audio", type="filepath")
        with gr.Row():
            mdx_button = gr.Button("Separate!", variant="primary")
        with gr.Row():
            mdx_stem1 = gr.Audio(label="Stem 1", type="filepath", interactive=False)
            mdx_stem2 = gr.Audio(label="Stem 2", type="filepath", interactive=False)

    with gr.Tab("VR ARCH"):
        with gr.Group():
            with gr.Row():
                vr_model = gr.Dropdown(label="Select the Model", choices=VR_ARCH_MODELS)
            with gr.Row():
                vr_window_size = gr.Slider(minimum=320, maximum=1024, step=32, value=512, label="Window Size", info="Balance quality and speed. 1024 = fast but lower, 320 = slower but better quality.")
                vr_aggression = gr.Slider(minimum=1, maximum=100, step=1, value=5, label="Agression", info="Intensity of primary stem extraction.")
                vr_tta = gr.Checkbox(value=False, label="TTA", info="Enable Test-Time-Augmentation; slow but improves quality.")
                vr_post_process = gr.Checkbox(value=False, label="Post Process", info="Identify leftover artifacts within vocal output; may improve separation for some songs.")
                vr_post_process_threshold = gr.Slider(minimum=0.1, maximum=0.3, step=0.1, value=0.2, label="Post Process Threshold", info="Threshold for post-processing.")
                vr_high_end_process = gr.Checkbox(value=False, label="High End Process", info="Mirror the missing frequency range of the output.")
        with gr.Row():
            vr_audio = gr.Audio(label="Input Audio", type="filepath")
        with gr.Row():
            vr_button = gr.Button("Separate!", variant="primary")
        with gr.Row():
            vr_stem1 = gr.Audio(label="Stem 1", type="filepath", interactive=False)
            vr_stem2 = gr.Audio(label="Stem 2", type="filepath", interactive=False)

    with gr.Tab("Demucs"):
        with gr.Group():
            with gr.Row():
                demucs_model = gr.Dropdown(label="Select the Model", choices=DEMUCS_MODELS)
            with gr.Row():
                demucs_seg_size = gr.Slider(minimum=1, maximum=100, step=1, value=40, label="Segment Size", info="Size of segments into which the audio is split. Higher = slower but better quality.")
                demucs_shifts = gr.Slider(minimum=0, maximum=20, step=1, value=2, label="Shifts", info="Number of predictions with random shifts, higher = slower but better quality.")
                demucs_overlap = gr.Slider(minimum=0.001, maximum=0.999, step=0.001, value=0.25, label="Overlap", info="Overlap between prediction windows. Higher = slower but better quality.")
                demucs_segments_enabled = gr.Checkbox(value=True, label="Segment-wise processing", info="Enable segment-wise processing.")
        with gr.Row():
            demucs_audio = gr.Audio(label="Input Audio", type="filepath")
        with gr.Row():
            demucs_button = gr.Button("Separate!", variant="primary")
        with gr.Row():
            demucs_stem1 = gr.Audio(label="Stem 1", type="filepath", interactive=False)
            demucs_stem2 = gr.Audio(label="Stem 2", type="filepath", interactive=False)
        with gr.Row():
            demucs_stem3 = gr.Audio(label="Stem 3", type="filepath", interactive=False)
            demucs_stem4 = gr.Audio(label="Stem 4", type="filepath", interactive=False)
        with gr.Row(visible=False) as stem6:
            demucs_stem5 = gr.Audio(label="Stem 5", type="filepath", interactive=False)
            demucs_stem6 = gr.Audio(label="Stem 6", type="filepath", interactive=False)

    with gr.Tab("Settings"):
        with gr.Accordion("General settings", open=False):
          with gr.Group():
              model_file_dir = gr.Textbox(value="/tmp/Audio-Separator-models/", label="Directory to cache model files", info="The directory where model files are stored.", placeholder="/tmp/Audio-Separator-models/")
              with gr.Row():
                  output_dir = gr.Textbox(value="output", label="File output directory", info="The directory where output files will be saved.", placeholder="output")
                  output_format = gr.Dropdown(value="wav", choices=["wav", "flac", "mp3"], label="Output Format", info="The format of the output audio file.")
              with gr.Row():
                  norm_threshold = gr.Slider(minimum=0.1, maximum=1, step=0.1, value=0.9, label="Normalization threshold", info="The threshold for audio normalization.")
                  amp_threshold = gr.Slider(minimum=0.0, maximum=1, step=0.1, value=0.0, label="Amplification threshold", info="The threshold for audio amplification.")
              with gr.Row():
                  batch_size = gr.Slider(minimum=1, maximum=16, step=1, value=1, label="Batch Size", info="Larger consumes more RAM but may process slightly faster.")

        with gr.Accordion("Rename Stems", open=False):
            gr.Markdown(
                """
                Keys for automatic determination of input file names, stems, and models to simplify the construction of output file names.
            
                Keys:
                * **NAME** - Input File Name
                * **STEM** - Stem Name (e.g., Vocals, Instrumental)
                * **MODEL** - Model Name (e.g., BS-Roformer-Viperx-1297)

                Example:
                * Usage: NAME_(STEM)_MODEL
                * Output File Name: Music_(Vocals)_BS-Roformer-Viperx-1297
                """
            )
            with gr.Row():
                vocals_stem = gr.Textbox(value="NAME_(STEM)_MODEL", label="Vocals Stem", info="Output example: Music_(Vocals)_BS-Roformer-Viperx-1297", placeholder="NAME_(STEM)_MODEL")
                instrumental_stem = gr.Textbox(value="NAME_(STEM)_MODEL", label="Instrumental Stem", info="Output example: Music_(Instrumental)_BS-Roformer-Viperx-1297", placeholder="NAME_(STEM)_MODEL")
                other_stem = gr.Textbox(value="NAME_(STEM)_MODEL", label="Other Stem", info="Output example: Music_(Other)_BS-Roformer-Viperx-1297", placeholder="NAME_(STEM)_MODEL")
            with gr.Row():
                drums_stem = gr.Textbox(value="NAME_(STEM)_MODEL", label="Drums Stem", info="Output example: Music_(Drums)_BS-Roformer-Viperx-1297", placeholder="NAME_(STEM)_MODEL")
                bass_stem = gr.Textbox(value="NAME_(STEM)_MODEL", label="Bass Stem", info="Output example: Music_(Bass)_BS-Roformer-Viperx-1297", placeholder="NAME_(STEM)_MODEL")
            with gr.Row():
                guitar_stem = gr.Textbox(value="NAME_(STEM)_MODEL", label="Guitar Stem", info="Output example: Music_(Guitar)_BS-Roformer-Viperx-1297", placeholder="NAME_(STEM)_MODEL")
                piano_stem = gr.Textbox(value="NAME_(STEM)_MODEL", label="Piano Stem", info="Output example: Music_(Piano)_BS-Roformer-Viperx-1297", placeholder="NAME_(STEM)_MODEL")

    demucs_model.change(update_stems, inputs=[demucs_model], outputs=stem6)

    roformer_button.click(
        roformer_separator,
        inputs=[
            roformer_audio,
            roformer_model,
            roformer_seg_size,
            roformer_override_seg_size,
            roformer_overlap,
            roformer_pitch_shift,
            model_file_dir,
            output_dir,
            output_format,
            norm_threshold,
            amp_threshold,
            batch_size,
            vocals_stem,
            instrumental_stem,
            other_stem,
            drums_stem,
            bass_stem,
            guitar_stem,
            piano_stem,
        ],
        outputs=[roformer_stem1, roformer_stem2],
    )
    mdx23c_button.click(
        mdx23c_separator,
        inputs=[
            mdx23c_audio,
            mdx23c_model,
            mdx23c_seg_size,
            mdx23c_override_seg_size,
            mdx23c_overlap,
            mdx23c_pitch_shift,
            model_file_dir,
            output_dir,
            output_format,
            norm_threshold,
            amp_threshold,
            batch_size,
            vocals_stem,
            instrumental_stem,
            other_stem,
            drums_stem,
            bass_stem,
            guitar_stem,
            piano_stem,
        ],
        outputs=[mdx23c_stem1, mdx23c_stem2],
    )
    mdx_button.click(
        mdx_separator,
        inputs=[
            mdx_audio,
            mdx_model,
            mdx_hop_length,
            mdx_seg_size,
            mdx_overlap,
            mdx_denoise,
            model_file_dir,
            output_dir,
            output_format,
            norm_threshold,
            amp_threshold,
            batch_size,
            vocals_stem,
            instrumental_stem,
            other_stem,
            drums_stem,
            bass_stem,
            guitar_stem,
            piano_stem,
        ],
        outputs=[mdx_stem1, mdx_stem2],
    )
    vr_button.click(
        vr_separator,
        inputs=[
            vr_audio,
            vr_model,
            vr_window_size,
            vr_aggression,
            vr_tta,
            vr_post_process,
            vr_post_process_threshold,
            vr_high_end_process,
            model_file_dir,
            output_dir,
            output_format,
            norm_threshold,
            amp_threshold,
            batch_size,
            vocals_stem,
            instrumental_stem,
            other_stem,
            drums_stem,
            bass_stem,
            guitar_stem,
            piano_stem,
        ],
        outputs=[vr_stem1, vr_stem2],
    )
    demucs_button.click(
        demucs_separator,
        inputs=[
            demucs_audio,
            demucs_model,
            demucs_seg_size,
            demucs_shifts,
            demucs_overlap,
            demucs_segments_enabled,
            model_file_dir,
            output_dir,
            output_format,
            norm_threshold,
            amp_threshold,
            vocals_stem,
            instrumental_stem,
            other_stem,
            drums_stem,
            bass_stem,
            guitar_stem,
            piano_stem,
        ],
        outputs=[demucs_stem1, demucs_stem2, demucs_stem3, demucs_stem4, demucs_stem5, demucs_stem6],
    )

def main():
    app.launch(share=True, debug=True)

if __name__ == "__main__":
    main()
