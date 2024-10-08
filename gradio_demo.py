import os
import gradio as gr
import yt_dlp
from audio_separator.separator import Separator

# Function to download audio from a link and return the file path
def download_audio_from_link(link):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': 'downloaded_audio.%(ext)s'
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([link])
    return "downloaded_audio.wav"

# Function for separating the audio
def separate_audio_from_link(link):
    # Step 1: Download the audio from the link
    input_file = download_audio_from_link(link)

    # Step 2: Set up output directory and models for separation
    output = "/content/output"
    separator = Separator(output_dir=output)

    # Define output paths
    vocals = os.path.join(output, 'Vocals.wav')
    instrumental = os.path.join(output, 'Instrumental.wav')
    vocals_reverb = os.path.join(output, 'Vocals (Reverb).wav')
    vocals_no_reverb = os.path.join(output, 'Vocals (No Reverb).wav')
    lead_vocals = os.path.join(output, 'Lead Vocals.wav')
    backing_vocals = os.path.join(output, 'Backing Vocals.wav')

    # Step 3: Splitting track into Vocal and Instrumental
    separator.load_model(model_filename='model_bs_roformer_ep_317_sdr_12.9755.ckpt')
    voc_inst = separator.separate(input_file)
    os.rename(os.path.join(output, voc_inst[0]), instrumental)
    os.rename(os.path.join(output, voc_inst[1]), vocals)

    # Step 4: Apply DeEcho-DeReverb to Vocals
    separator.load_model(model_filename='UVR-DeEcho-DeReverb.pth')
    voc_no_reverb = separator.separate(vocals)
    os.rename(os.path.join(output, voc_no_reverb[0]), vocals_no_reverb)
    os.rename(os.path.join(output, voc_no_reverb[1]), vocals_reverb)

    # Step 5: Separate Back Vocals from Main Vocals
    separator.load_model(model_filename='mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt')
    backing_voc = separator.separate(vocals_no_reverb)
    os.rename(os.path.join(output, backing_voc[0]), backing_vocals)
    os.rename(os.path.join(output, backing_voc[1]), lead_vocals)

    # Return paths to the processed files
    return instrumental, vocals, vocals_no_reverb, vocals_reverb, lead_vocals, backing_vocals

# Define the Gradio Interface
with gr.Blocks(theme="NoCrypt/miku@1.2.2", title="Audio Separator Demo") as demo:
    gr.Markdown("# Audio Separator Gradio demo")

    with gr.Row():
        with gr.Column():
            link_input = gr.Textbox(label="Enter Audio/Video Link")
            separate_button = gr.Button("Download and Separate Audio")
        
        with gr.Column():
            instrumental_output = gr.Audio(label="Instrumental Output")
            vocals_output = gr.Audio(label="Vocals Output")
            vocals_no_reverb_output = gr.Audio(label="Vocals No Reverb Output")
            vocals_reverb_output = gr.Audio(label="Vocals Reverb Output")
            lead_vocals_output = gr.Audio(label="Lead Vocals Output")
            backing_vocals_output = gr.Audio(label="Backing Vocals Output")

    # Define button functionality
    separate_button.click(
        separate_audio_from_link,
        inputs=[link_input],
        outputs=[
            instrumental_output,
            vocals_output,
            vocals_no_reverb_output,
            vocals_reverb_output,
            lead_vocals_output,
            backing_vocals_output
        ]
    )

# Launch the Gradio app
demo.launch(debug=True, share=True)
