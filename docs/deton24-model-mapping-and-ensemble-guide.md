# Deton24 Doc ↔ Audio-Separator Model Mapping & Ensemble Guide

**Date:** 2026-03-15
**Source document:** `docs/deton24-audio-separation-info-2026-03-15.md` (converted from [deton24's Google Doc](https://docs.google.com/document/d/17fjNvJzj8ZGSer7c7OFe_CNfUKbAxEh_OBv94ZdRG5c))

## Purpose

This document serves as:
1. A **naming convention lookup table** to map between audio-separator model filenames and the informal names used in deton24's doc and the audio separation community
2. A **section reference guide** with line numbers into the deton24 doc for finding specific topics
3. A **task brief for Claude agents** implementing improvements to audio-separator (ensemble presets, new models, phase fix)

---

## Table of Contents

- [Section 1: How to Navigate the Deton24 Doc](#section-1-how-to-navigate-the-deton24-doc)
- [Section 2: Naming Convention Lookup Table](#section-2-naming-convention-lookup-table)
- [Section 3: Key Metrics Explained](#section-3-key-metrics-explained)
- [Section 4: Ensemble Algorithm Mapping](#section-4-ensemble-algorithm-mapping)
- [Section 5: Recommended Ensemble Presets (Implementable Now)](#section-5-recommended-ensemble-presets-implementable-now)
- [Section 6: Missing Top-Tier Models to Add](#section-6-missing-top-tier-models-to-add)
- [Section 7: Phase Fix — What It Is and How to Implement It](#section-7-phase-fix--what-it-is-and-how-to-implement-it)
- [Section 8: Agent Task Briefs](#section-8-agent-task-briefs)

---

## Section 1: How to Navigate the Deton24 Doc

The deton24 doc (`docs/deton24-audio-separation-info-2026-03-15.md`) is ~27,000+ lines. Here are the key section locations:

| Topic | Approx. Line | Section Heading |
|-------|-------------|-----------------|
| **Best models list** | 6845 | `### **The best models**` |
| **Best instrumental models** | 6858 | `###### > for instrumentals` |
| **Instrumental ensembles** | 7676 | `###### **>Ensembles**` (for instrumentals) |
| **Best vocal models** | 8352 | `###### **>** ***for vocals***` |
| **Vocal ensembles** | 8769 | `###### **Ensembles**` (for vocals) |
| **Debleeding/cleaning** | 9421 | `###### **Debleeding/cleaning vocals/instrumentals/inverts**` |
| **Karaoke** | 9636 | `###### **>Karaoke**` |
| **Lead vocals only** | 10034 | `###### >Keeping only **lead vocals**` |
| **Drumsep** | 10717 | `###### **>Sep. parts of drums a.k.a. Drumsep**` |
| **De-reverb** | 11220 | `###### **De-reverb**` |
| **De-noising** | 11538 | `###### **De-noising (vinyl noise/white noise/general)**` |
| **Ensemble algorithm explanations** | 12212 | `###### *Ensemble algorithm explanations*` |
| **4-5 max models rule** | 12351 | `###### *4-5 max ensemble models rule*` |
| **SDR leaderboard** | 13286 | `##### SDR leaderboard` |
| **Instrumental bleedless rankings** | 13478 | `##### *Instrumental models sorted by instrumental* ***bleedless*** *metric:*` |
| **Vocal bleedless rankings** | 13558 | `###### Vocal models/ensembles sorted by instrumental **bleedless** metric` |
| **Phantom center extraction** | 14731 | `##### Similarity/Phantom Center/Mid channel Extractor` |
| **Drumsep (detailed)** | 18938 | `## Drumsep - single percussion instruments separation` |
| **Phase fixer/swapper** | 27072 | `###### *Phase fixer/swapper*` |

### Tips for navigating

- The doc uses **community nicknames**, not filenames. E.g., "v1e+" = `melband_roformer_inst_v1e_plus.ckpt`, "deux" = becruily's dual vocal+instrumental Mel-Roformer, "Resurrection" = unwa's BS-Roformer models.
- Model creators are referenced by their Discord handles: **unwa** (pcunwa), **Gabox** (GaboxR67), **becruily**, **aufr33**, **viperx**, **anvuew**, **jarredou**, **Aname** (Aname-Tommy), **mesk** (meskvlla33), **ZFTurbo** (MVSEP developer).
- "MVSEP exclusive" means the model is only available via mvsep.com and cannot be downloaded — these cannot be added to audio-separator.
- "UVR" = Ultimate Vocal Remover GUI, the main desktop app most community members use.
- "MSST" = Music Source Separation Training, the framework used to train and run inference on newer Roformer models.

---

## Section 2: Naming Convention Lookup Table

### Roformer — Vocals

| Community Name | audio-separator filename | Creator | Key Metrics | Notes |
|---|---|---|---|---|
| Kim (original) | `vocals_mel_band_roformer.ckpt` | KimberleyJSN | — | The original Mel-Roformer |
| Kim FT | `mel_band_roformer_kim_ft_unwa.ckpt` | unwa | — | Fine-tune of Kim |
| Kim FT2 | `mel_band_roformer_kim_ft2_unwa.ckpt` | unwa | — | |
| FT2 bleedless | `mel_band_roformer_kim_ft2_bleedless_unwa.ckpt` | unwa | bleedless 39.30 | Very clean vocals |
| Kim FT3 (preview) | `mel_band_roformer_kim_ft3_unwa.ckpt` | unwa | — | Used as phase fix reference in some setups |
| becruily vocal | `mel_band_roformer_vocals_becruily.ckpt` | becruily | — | Key phase-fix reference model; part of "deux" dual |
| Gabox voc | `mel_band_roformer_vocals_gabox.ckpt` | Gabox | — | |
| Gabox voc v2 | `mel_band_roformer_vocals_v2_gabox.ckpt` | Gabox | — | |
| Gabox voc fv1–fv6 | `mel_band_roformer_vocals_fv1_gabox.ckpt` … `fv6` | Gabox | fv4 best for RVC | fv6 = extreme fullness (24.93) |
| Big Beta 4 | `melband_roformer_big_beta4.ckpt` | unwa | — | |
| Big Beta 5e | `melband_roformer_big_beta5e.ckpt` | unwa | — | |
| Big Beta 6 | `melband_roformer_big_beta6.ckpt` | unwa | — | |
| Big Beta 6X | `melband_roformer_big_beta6x.ckpt` | unwa | SDR 11.12 | Good balance |
| Revive | `bs_roformer_vocals_revive_unwa.ckpt` | unwa | — | |
| Revive 2 | `bs_roformer_vocals_revive_v2_unwa.ckpt` | unwa | bleedless 40.07 | Highest bleedless vocal |
| Revive 3e | `bs_roformer_vocals_revive_v3e_unwa.ckpt` | unwa | fullness 21.43 | High fullness vocal |
| Resurrection (vocal) | `bs_roformer_vocals_resurrection_unwa.ckpt` | unwa | SDR 11.34, bleedless 39.99 | Top-tier, only 195MB, fast |
| FullnessVocalModel | `mel_band_roformer_vocal_fullness_aname.ckpt` | Aname | — | |
| SYHFT v1-v3 | `MelBandRoformerSYHFT.ckpt` etc. | — | — | Not prominently discussed in deton24 |

### Roformer — Instrumentals

| Community Name | audio-separator filename | Creator | Key Metrics | Notes |
|---|---|---|---|---|
| becruily inst | `mel_band_roformer_instrumental_becruily.ckpt` | becruily | SDR 17.55, bleedless 41.36 | Part of "deux" dual; "SOTA" per community |
| Gabox inst / inst2 / inst3 | `mel_band_roformer_instrumental_gabox.ckpt` etc. | Gabox | — | |
| Gabox bleedless v1–v3 | `mel_band_roformer_instrumental_bleedless_v1_gabox.ckpt` etc. | Gabox | — | Optimized for low bleed |
| Gabox fullness v1–v3 | `mel_band_roformer_instrumental_fullness_v1_gabox.ckpt` etc. | Gabox | — | Optimized for fullness |
| Gabox fullness noise v4 | `mel_band_roformer_instrumental_fullness_noise_v4_gabox.ckpt` | Gabox | fullness 40.40 | a.k.a. "inst_Fv4Noise" |
| INSTV5 / INSTV5N | `mel_band_roformer_instrumental_instv5_gabox.ckpt` / `instv5n` | Gabox | — | |
| INSTV6 / INSTV6N | `mel_band_roformer_instrumental_instv6_gabox.ckpt` / `instv6n` | Gabox | INSTV6N: fullness 41.68 | Extreme fullness, noisy |
| INSTV7 / INSTV7N | `mel_band_roformer_instrumental_instv7_gabox.ckpt` / `instv7n` | Gabox | — | |
| INSTV8 / INSTV8N | `mel_band_roformer_instrumental_instv8_gabox.ckpt` / `instv8n` | Gabox | — | |
| Inst_GaboxFv7z | `mel_band_roformer_instrumental_fv7z_gabox.ckpt` | Gabox | bleedless 44.61 | **Best bleedless inst**, nearly noiseless |
| Inst_GaboxFv8 | `mel_band_roformer_instrumental_fv8_gabox.ckpt` | Gabox | — | |
| Inst_GaboxFVX | `mel_band_roformer_instrumental_fvx_gabox.ckpt` | Gabox | — | |
| v1e | `melband_roformer_inst_v1e.ckpt` | unwa | fullness 38.87 | Classic fullness model |
| v1e+ | `melband_roformer_inst_v1e_plus.ckpt` | unwa | fullness 37.89 | Best balanced fullness |
| v1+ | `melband_roformer_inst_v1_plus.ckpt` | unwa | — | |
| Resurrection Inst | `bs_roformer_instrumental_resurrection_unwa.ckpt` | unwa | SDR 17.25 | Only 200MB, great all-rounder |
| BS-Roformer SW | `BS-Roformer-SW.ckpt` | — | — | Reversed Apple Logic Pro, 6-stem |

### Roformer — Karaoke

| Community Name | audio-separator filename | Creator | Key Metrics |
|---|---|---|---|
| aufr33/viperx karaoke | `mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt` | aufr33/viperx | SDR 10.20 |
| kar_gabox | `mel_band_roformer_karaoke_gabox.ckpt` | Gabox | — |
| Karaoke_GaboxV2 | `mel_band_roformer_karaoke_gabox_v2.ckpt` | Gabox | — |
| becruily karaoke | `mel_band_roformer_karaoke_becruily.ckpt` | becruily | — |

### Roformer — De-reverb / Denoise / Special

| Community Name | audio-separator filename | Creator | Notes |
|---|---|---|---|
| Aufr33 Denoise (average) | `denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt` | aufr33 | Recommended first-choice denoiser |
| Aufr33 Denoise (aggressive) | `denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt` | aufr33 | |
| Gabox denoisedebleed | `mel_band_roformer_denoise_debleed_gabox.ckpt` | Gabox | Preprocessor |
| anvuew dereverb (stereo) | `dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt` | anvuew | |
| anvuew dereverb (less aggressive) | `dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt` | anvuew | |
| anvuew dereverb mono | `dereverb_mel_band_roformer_mono_anvuew.ckpt` | anvuew | SDR 20.40 |
| crowd removal | `mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt` | aufr33/viperx | |
| bleed suppressor | `mel_band_roformer_bleed_suppressor_v1.ckpt` | unwa/97chris | For instrumentals post-processing |
| chorus separator | `model_chorus_bs_roformer_ep_267_sdr_24.1275.ckpt` | — | |
| aspiration/breath models | `aspiration_mel_band_roformer_sdr_18.9845.ckpt` etc. | — | |

### MDX / MDX23C / VR / Demucs

| Community Name | audio-separator filename | Notes |
|---|---|---|
| UVR-MDX-NET-Inst_HQ_5 | `UVR-MDX-NET-Inst_HQ_5.onnx` | Older MDX arch, still useful for fast/low-resource ensemble |
| MDX23C DrumSep | `MDX23C-DrumSep-aufr33-jarredou.ckpt` | 5-stem drum separation (kick/snare/toms/cymbals/other) |
| MDX23C De-Reverb | `MDX23C-De-Reverb-aufr33-jarredou.ckpt` | |
| UVR De-Reverb | `UVR-De-Reverb-aufr33-jarredou.pth` | VR arch |
| BVE (Backing Vocal Extractor) | `UVR-BVE-4B_SN-44100-2.pth` | VR arch, extracts backing vocals |
| htdemucs_ft | `htdemucs_ft.yaml` | 4-stem (vocals/drums/bass/other), recommended for drum pre-separation |

---

## Section 3: Key Metrics Explained

The community uses three key metrics to evaluate models (see deton24 doc ~line 6850 and the leaderboard at ~line 13286):

| Metric | What It Measures | Higher = |
|--------|-----------------|----------|
| **SDR** (Signal-to-Distortion Ratio) | Overall separation quality vs. ground truth | Better overall quality |
| **Bleedless** | How little the *other* stems bleed into the target stem | Cleaner output, but potentially muddier/more muted |
| **Fullness** | How much of the target stem's content is preserved | Fuller/richer output, but potentially noisier with more bleed |

**Important:** Bleedless and fullness are in tension. A model cannot maximize both. The community categorizes models as:
- **Fullness models**: v1e, INSTV6N, inst_Fv4Noise — preserve more instruments but have noise/vocal residues
- **Bleedless models**: Fv7z, FNO, Revive 2 — cleaner but may lose some subtle instruments
- **Balanced models**: Resurrection Inst, becruily "deux", Beta 6X — attempt to optimize both

The best ensembles typically combine a fullness model with a bleedless model using an appropriate algorithm.

---

## Section 4: Ensemble Algorithm Mapping

The deton24 doc (line 12212–12365) explains ensemble algorithms. Here's how the community terminology maps to audio-separator's `Ensembler` class (`audio_separator/separator/ensembler.py`):

| Community / UVR Term | audio-separator algorithm | Effect | Best For |
|---|---|---|---|
| **Avg Spec** / **Average** | `avg_wave` or `avg_fft` | Averages all inputs | Highest SDR, safest default |
| **Max Spec** | `uvr_max_spec` or `max_fft` | Keeps loudest frequency bins | Fuller output, more bleed; good for vocals |
| **Min Spec** | `uvr_min_spec` or `min_fft` | Keeps quietest frequency bins | Cleaner output, less bleed; good for instrumentals |
| **Max FFT** | `max_fft` | Same concept as Max Spec in FFT domain | Used interchangeably with Max Spec in community |
| **Min FFT** | `min_fft` | Same concept as Min Spec in FFT domain | |
| **Median** | `median_wave` or `median_fft` | Takes median value | Reduces outlier artifacts |

### Community rules of thumb (from deton24 line 12212+):
- **Avg/Avg** gets the highest SDR and is the safest default
- **Max Spec** for vocals (fuller, captures more vocal content, more instrument bleed)
- **Min Spec** for instrumentals (cleaner, less vocal residue, but can sound muffled)
- **Max/Min** = max for vocal stem, min for instrumental stem (commonly recommended)
- **Do not ensemble more than 4-5 models** — SDR drops above this (line 12351)

---

## Section 5: Recommended Ensemble Presets (Implementable Now)

These ensembles use **only models already in audio-separator**. Ranked by community consensus from the deton24 doc.

### Instrumental Ensembles

#### Preset: `instrumental_clean` — Bleedless + All-Rounder
```
Models:
  - mel_band_roformer_instrumental_fv7z_gabox.ckpt
  - bs_roformer_instrumental_resurrection_unwa.ckpt
Algorithm: uvr_max_spec
```
**Why:** Fv7z is the bleedless king (44.61) and Resurrection Inst is a great all-rounder (SDR 17.25). Max Spec fills in what Fv7z might miss. Deton24 doc line ~7739 discusses Fv7z in ensembles.

#### Preset: `instrumental_full` — Maximum Instrument Preservation
```
Models:
  - melband_roformer_inst_v1e_plus.ckpt
  - mel_band_roformer_instrumental_becruily.ckpt
Algorithm: uvr_max_spec
```
**Why:** v1e+ (fullness 37.89) is the community's classic fullness model. Becruily inst is "SOTA" (SDR 17.55). Max Spec preserves energy. Based on ensemble patterns from deton24 line ~7777-7789.

#### Preset: `instrumental_balanced` — Good Balance of Noise/Fullness
```
Models:
  - mel_band_roformer_instrumental_instv8_gabox.ckpt
  - bs_roformer_instrumental_resurrection_unwa.ckpt
Algorithm: uvr_max_spec
```
**Why:** Inspired by deton24 line ~7743 ("Gabox Inst V8 + [model] = good balance between noise and fullness"). Uses Resurrection Inst as the secondary model since the MVSEP model referenced there isn't available locally.

#### Preset: `instrumental_low_resource` — Fast / Low VRAM
```
Models:
  - bs_roformer_instrumental_resurrection_unwa.ckpt
  - UVR-MDX-NET-Inst_HQ_5.onnx
Algorithm: avg_fft
```
**Why:** Resurrection Inst is only 200MB and fast. HQ_5 is MDX arch (very fast). Avg is safest. Deton24 line ~7855 mentions HQ_5 in low-resource ensembles.

### Vocal Ensembles

#### Preset: `vocal_balanced` — Best Overall Quality
```
Models:
  - bs_roformer_vocals_resurrection_unwa.ckpt
  - melband_roformer_big_beta6x.ckpt
Algorithm: avg_fft
```
**Why:** Resurrection (SDR 11.34, bleedless 39.99) + Beta 6X (SDR 11.12). Two top-tier models averaged for highest SDR. Community recommendations at deton24 line ~8789-8791.

#### Preset: `vocal_clean` — Minimal Instrument Bleed
```
Models:
  - bs_roformer_vocals_revive_v2_unwa.ckpt
  - mel_band_roformer_kim_ft2_bleedless_unwa.ckpt
Algorithm: min_fft
```
**Why:** Revive 2 (bleedless 40.07) + FT2 bleedless (39.30) = cleanest possible. Min FFT removes anything not common to both, further reducing bleed.

#### Preset: `vocal_full` — Maximum Vocal Capture
```
Models:
  - bs_roformer_vocals_revive_v3e_unwa.ckpt
  - mel_band_roformer_vocals_becruily.ckpt
Algorithm: max_fft
```
**Why:** Revive 3e (fullness 21.43) + becruily vocal (fullness 23.25). Max FFT keeps all vocal content from both. Good for capturing harmonies and backing vocals. Related: deton24 line ~8783-8795.

#### Preset: `vocal_rvc` — Optimized for RVC/AI Training
```
Models:
  - melband_roformer_big_beta6x.ckpt
  - mel_band_roformer_vocals_fv4_gabox.ckpt
Algorithm: avg_wave
```
**Why:** Directly recommended at deton24 line ~8789-8791 for RVC: "beta6x + voc_fv4". Average for clean, consistent output suitable for training data.

### Karaoke Ensembles

#### Preset: `karaoke` — Lead Vocal Removal (3-model)
```
Models:
  - mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt
  - mel_band_roformer_karaoke_gabox_v2.ckpt
  - mel_band_roformer_karaoke_becruily.ckpt
Algorithm: avg_wave
```
**Why:** Deton24 reports 3-model karaoke ensembles reach SDR ~10.6 vs ~10.2 for single models. These are the three main karaoke models available in audio-separator.

### Processing Pipelines (Sequential, Not Parallel Ensemble)

#### Pipeline: `clean_vocals` — Full Vocal Cleaning Chain
```
Step 1: Separate vocals (vocal_balanced preset or single model)
Step 2: denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt (denoise)
Step 3: dereverb_mel_band_roformer_mono_anvuew.ckpt (dereverb)
```
**Why:** Deton24 line ~8807-8817 recommends: vocals → de-reverb → karaoke → de-noise. The denoise-first approach is also recommended by community member natethegratevhs.

#### Pipeline: `drumsep` — Detailed Drum Separation
```
Step 1: htdemucs_ft.yaml (extract drums stem from mix)
Step 2: MDX23C-DrumSep-aufr33-jarredou.ckpt (split drums into kick/snare/toms/cymbals)
```
**Why:** Deton24 line ~10717+ recommends pre-separating drums with htdemucs_ft, then using drumsep for sub-stem separation. The jarredou MDX23C model is the only downloadable drumsep model.

---

## Section 6: Missing Top-Tier Models to Add

These models are discussed prominently in the deton24 doc, are publicly downloadable, and would significantly improve audio-separator's quality — especially for ensembles.

### High Priority (top-tier, public, frequently referenced)

| Model | Creator | Type | HuggingFace URL | Key Metrics | Deton24 Line |
|---|---|---|---|---|---|
| BS-Roformer HyperACE v2 inst | unwa | instrumental | `pcunwa/BS-Roformer-HyperACE` (check exact path) | SDR 17.40, fullness 38.03 | ~7682, 7698, 7702 |
| BS-Roformer HyperACE v2 voc | unwa | vocal | same repo | SDR 11.40 | ~8777 |
| BS-Roformer-Inst-FNO | unwa | instrumental | `pcunwa/` (check) | SDR 17.60 (highest public inst) | ~7714, 7755 |
| Mel "deux" (becruily) | becruily | dual (vocal+inst) | `becruily/mel-band-roformer-deux` | inst SDR 17.55, voc SDR 11.37 | ~7682, 7698, 8773 |
| Big Beta 7 | unwa | vocal | `pcunwa/` (check) | SDR 11.20 | ~8775 |
| Gabox voc_fv7 + betas | Gabox | vocal | `GaboxR67/MelBandRoformers` | SDR 11.16 | ~8826 |
| Gabox inst_gaboxFlowersV10 | Gabox | instrumental | same repo | SDR 16.95, fullness 37.12 | ~7759 |
| Rifforge | mesk | instrumental (metal) | `meskvlla33/rifforge` | — | ~7759 |
| BS-Roformer 1296/1297 | viperx | vocal | check HF | SDR 12.96/12.97 | ~7753, 7847-7852 |
| anvuew dereverb BS (stereo, SDR 22.50) | anvuew | de-reverb | check HF | SDR 22.50 | ~100 |
| BS_RoFormer_mag | anvuew | vocal | `anvuew/BS_RoFormer_mag` | bleedless 32.17, fullness 22.15 | ~59 |

### Medium Priority (useful but less critical)

| Model | Creator | Type | Notes |
|---|---|---|---|
| BS-Roformer-Inst-EXP-Value-Residual | unwa | instrumental | Experimental |
| BS-EXP-SiameseRoformer | unwa | vocal | Experimental, needs custom code |
| gilliaan MonoStereo Dual Beta1/2 | gilliaan | phantom center | Niche use case |
| Neo_InstVFX | neoculture | instrumental | Preserves vocal chops (K-pop) |

### Cannot Add (MVSEP Exclusive)

These are referenced constantly in the doc but **cannot be downloaded** — they only work through mvsep.com:

- BS-Roformer 2025.07, 2025.06, 2024.08, 2024.04 (by ZFTurbo)
- SCNet XL, SCNet Large, SCNet XL IHF, SCNet variants
- Various instrument-specific models (saxophone, trumpet, violin, etc.)
- VitLarge23
- Drumsep 4/5/6 stem Mel-Roformer and SCNet models

---

## Section 7: Phase Fix — What It Is and How to Implement It

### Background

Phase fix (also called "phase swapper") was invented by **aufr33** and is discussed in detail at deton24 doc line 27072-27170. It is one of the most impactful post-processing techniques for instrumental separation.

### The Problem It Solves

Roformer models trained with an **instrumental stem target** produce full-sounding instrumentals but with noisy vocal residues. Models trained with a **vocal stem target** produce muddy instrumentals (via inversion) but with better bleedless metrics. Phase fix combines the best of both.

### How It Works (Technical)

From jarredou's explanation (deton24 line 27160-27164):

> "There is a phase in the STFT domain. For each STFT bin, it will use the **magnitude data** of one model [the instrumental model], and the **phase data** from another model [the vocal model's instrumental output], and hopefully this can improve the final result."

From santilli_ (line 27166):

> "What the script does is blend the STFT phase of the 'donor' file into the target. It uses different blending scales for different frequencies so that it'll only affect the parts that are directly related to the perceived noise."

### Algorithm (pseudocode)

```python
import librosa
import numpy as np

def phase_fix(target_audio, reference_audio, sr,
              low_cutoff=500, high_cutoff=5000,
              high_freq_weight=0.8, n_fft=2048):
    """
    Apply phase fix to target audio using reference audio's phase.

    Args:
        target_audio: The instrumental from an inst-trained model (full but noisy)
        reference_audio: The instrumental from a vocal-trained model (muddy but clean phase)
        low_cutoff: Frequency below which phase is not replaced (Hz)
        high_cutoff: Frequency above which phase blending weight decreases (Hz)
        high_freq_weight: Blending weight for frequencies above high_cutoff (0.8-2.0)
        n_fft: FFT size
    Returns:
        Phase-fixed audio
    """
    # Convert to STFT
    target_stft = librosa.stft(target_audio, n_fft=n_fft)
    reference_stft = librosa.stft(reference_audio, n_fft=n_fft)

    # Get magnitude from target, phase from reference
    target_magnitude = np.abs(target_stft)
    reference_phase = np.angle(reference_stft)
    target_phase = np.angle(target_stft)

    # Create frequency-dependent blending weights
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    blend_weights = np.zeros_like(freqs)

    for i, f in enumerate(freqs):
        if f < low_cutoff:
            blend_weights[i] = 0.0  # Keep target phase below low_cutoff
        elif f < high_cutoff:
            # Linear interpolation between cutoffs
            blend_weights[i] = (f - low_cutoff) / (high_cutoff - low_cutoff)
        else:
            blend_weights[i] = high_freq_weight  # Use high_freq_weight above high_cutoff

    # Apply blending: new_phase = target_phase * (1 - weight) + reference_phase * weight
    blend_weights = blend_weights[:, np.newaxis]  # Shape for broadcasting
    blended_phase = target_phase * (1 - blend_weights) + reference_phase * blend_weights

    # Reconstruct with target magnitude and blended phase
    fixed_stft = target_magnitude * np.exp(1j * blended_phase)
    fixed_audio = librosa.istft(fixed_stft, length=len(target_audio))

    return fixed_audio
```

### Key Parameters (from community recommendations)

| Parameter | Default (UVR) | Default (Colab) | santilli_ recommendation | Use Case |
|---|---|---|---|---|
| `low_cutoff` | 500 Hz | 500 Hz | 100–420 Hz | Lower = more aggressive fix |
| `high_cutoff` | 5000 Hz | 9000 Hz | 100–4200 Hz | Lower = more aggressive fix |
| `high_freq_weight` | 0.8 | 0.8 | **2.0** | "Increasing from 0.8 to 2 is beneficial" (line 27076) |

### Common Phase Fix Pairings

These are the recommended target→reference pairings from the doc (line 27086-27098):

| Target (Instrumental Model) | Reference (Vocal/Phase-donor Model) | Cutoff Values |
|---|---|---|
| `melband_roformer_inst_v1e.ckpt` | `mel_band_roformer_vocals_becruily.ckpt` (inst output) | 100/100 (aggressive) |
| `melband_roformer_inst_v1e_plus.ckpt` | `mel_band_roformer_vocals_becruily.ckpt` (inst output) | 100/100 |
| `bs_roformer_instrumental_resurrection_unwa.ckpt` | `mel_band_roformer_vocals_becruily.ckpt` (inst output) | 3000/5000 |
| Any Gabox inst model | `mel_band_roformer_vocals_becruily.ckpt` (inst output) | 500/5000 |

**Important:** The "reference" is the **instrumental output** of a vocal-targeted model (not the vocal output). You separate with the vocal model, take its instrumental stem, and use that as the phase reference.

### Existing Code References

- **Becruily's original scripts:** https://drive.google.com/drive/folders/1JOa198ALJ0SnEreCq2y2kVj-sktvPePy
- **Phase Fixer Colab (santilli_):** https://colab.research.google.com/drive/1PMQmFRZb_XRIKnBjXhYlxNlZ5XcKMWXm
- **MSST repo** has phase fix integrated into inference pipeline

### Implementation Suggestion for audio-separator

Phase fix should be added as a post-processing step in the `Ensembler` class or as a standalone utility. The flow would be:

1. User runs separation with an instrumental model → gets `instrumental_target.wav`
2. User runs separation with a vocal model → gets `instrumental_reference.wav` (the "other" stem)
3. Phase fix combines magnitude from step 1 with phase from step 2
4. Optionally, the phase-fixed result is then ensembled with other models' outputs

For presets, this could be exposed as a pipeline configuration:
```yaml
preset: instrumental_phase_fixed
steps:
  - model: melband_roformer_inst_v1e_plus.ckpt
    output: target_instrumental
  - model: mel_band_roformer_vocals_becruily.ckpt
    output: reference_instrumental  # use the instrumental stem
  - phase_fix:
      target: target_instrumental
      reference: reference_instrumental
      low_cutoff: 100
      high_cutoff: 100
      high_freq_weight: 2.0
```

---

## Section 8: Agent Task Briefs

### Task A: Implement Ensemble Preset Configurations

**Goal:** Add preset ensemble configurations to audio-separator that users can invoke by name.

**Key files to modify:**
- `audio_separator/separator/ensembler.py` — already has ensemble algorithms implemented
- `audio_separator/separator/separator.py` — main separator class, handles model loading and separation
- `audio_separator/utils/cli.py` — CLI interface, needs preset selection flags
- A new config file (e.g., `audio_separator/ensemble_presets.json` or similar) for preset definitions

**Presets to implement:** See [Section 5](#section-5-recommended-ensemble-presets-implementable-now) above. Start with these core presets:
1. `instrumental_clean` — Fv7z + Resurrection Inst (max_spec)
2. `instrumental_full` — v1e+ + becruily inst (max_spec)
3. `vocal_balanced` — Resurrection voc + Beta 6X (avg_fft)
4. `vocal_clean` — Revive 2 + FT2 bleedless (min_fft)
5. `karaoke` — 3-model karaoke ensemble (avg_wave)

**UX suggestion:** `audio-separator input.wav --ensemble instrumental_clean`

**Reference:** Read the existing ensemble tests at `tests/unit/test_ensembler.py` for the current test patterns.

### Task B: Add Support for Missing Top-Tier Models

**Goal:** Add the high-priority models from [Section 6](#section-6-missing-top-tier-models-to-add) to audio-separator.

**Key files to modify:**
- `audio_separator/models.json` — model registry with download URLs and config
- `audio_separator/separator/separator.py` — `list_supported_model_files()` method (lines ~440-608)

**Process for each model:**
1. Find the model on HuggingFace (check creator repos listed in Section 6)
2. Identify the correct YAML config file (most Roformers need a paired YAML)
3. Add entry to `models.json` with the download URL and config
4. Test that the model loads and produces output
5. Verify the output stem names match expectations

**Priority order:**
1. BS-Roformer HyperACE v2 (inst + voc) — used in current best ensembles
2. Mel "deux" by becruily — SOTA instrumental, "doesn't need phase fix"
3. BS-Roformer-Inst-FNO — highest public inst SDR
4. Big Beta 7 — latest vocal model
5. Gabox voc_fv7 + inst_gaboxFlowersV10 — latest Gabox models
6. Rifforge — metal-specific, niche but popular
7. BS-Roformer 1296/1297 (viperx) — classic models, still used for phase fix

**HuggingFace repos to check:**
- `pcunwa/` (unwa's models)
- `GaboxR67/MelBandRoformers` (Gabox models)
- `becruily/mel-band-roformer-deux` (deux dual model)
- `meskvlla33/rifforge` (Rifforge)
- `anvuew/` (anvuew models)

### Task C: Implement Phase Fix Support

**Goal:** Add phase fix as a post-processing step in audio-separator.

**What to read first:**
- [Section 7](#section-7-phase-fix--what-it-is-and-how-to-implement-it) of this document (algorithm, parameters, pseudocode)
- Deton24 doc line 27072-27170 for full community discussion
- Becruily's scripts: https://drive.google.com/drive/folders/1JOa198ALJ0SnEreCq2y2kVj-sktvPePy
- Phase Fixer Colab source: https://colab.research.google.com/drive/1PMQmFRZb_XRIKnBjXhYlxNlZ5XcKMWXm

**Implementation approach:**

1. **Add a `PhaseFixer` class** (new file, e.g., `audio_separator/separator/phase_fixer.py`):
   - Takes target audio (from inst model) and reference audio (from vocal model's inst stem)
   - Parameters: `low_cutoff`, `high_cutoff`, `high_freq_weight`, `n_fft`
   - Works in STFT domain: keeps target magnitude, blends in reference phase
   - Must handle stereo (process each channel independently)

2. **Integrate with Separator**:
   - Add `--phase_fix_reference_model` CLI flag
   - When set, separator runs the reference model first, keeps its instrumental stem
   - Then runs the target model, applies phase fix using the reference instrumental
   - Outputs the phase-fixed instrumental

3. **Integrate with ensemble presets**:
   - Some presets should include phase fix as a step (see Section 5 pipelines)
   - The preset config format should support multi-step pipelines

**Key gotcha:** The "reference" for phase fix is the **instrumental stem from a vocal-trained model** (not the vocal stem itself). So if using `mel_band_roformer_vocals_becruily.ckpt` as reference, you need its "Instrumental" / "No Vocals" output, not its "Vocals" output.

**Testing:** Compare output of phase-fixed v1e (using becruily vocal as reference, cutoffs 100/100) against plain v1e output. The phase-fixed version should have noticeably less vocal residue/"noise" in quiet passages while maintaining instrument fullness.
