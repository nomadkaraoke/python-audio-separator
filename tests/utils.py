import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import soundfile as sf
from pathlib import Path


def generate_waveform_image(audio_path, output_path=None, fig_size=(10, 4)):
    """Generate a waveform image from an audio file.
    
    Args:
        audio_path: Path to the audio file
        output_path: Path to save the generated image (optional)
        fig_size: Size of the figure (width, height)
        
    Returns:
        BytesIO object containing the image if output_path is None, otherwise saves to output_path
    """
    # Load audio file
    y, sr = librosa.load(audio_path, sr=None, mono=False)
    
    # If mono, convert to stereo-like format for consistent plotting
    if y.ndim == 1:
        y = np.array([y, y])
    
    plt.figure(figsize=fig_size)
    
    # Plot waveform for each channel
    plt.subplot(2, 1, 1)
    plt.plot(y[0])
    plt.title('Channel 1')
    plt.subplot(2, 1, 2)
    plt.plot(y[1])
    plt.title('Channel 2')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
        return output_path
    else:
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return buf


def generate_spectrogram_image(audio_path, output_path=None, fig_size=(10, 8)):
    """Generate a spectrogram image from an audio file.
    
    Args:
        audio_path: Path to the audio file
        output_path: Path to save the generated image (optional)
        fig_size: Size of the figure (width, height)
        
    Returns:
        BytesIO object containing the image if output_path is None, otherwise saves to output_path
    """
    # Load audio file
    y, sr = librosa.load(audio_path, sr=None, mono=False)
    
    # If mono, convert to stereo-like format for consistent plotting
    if y.ndim == 1:
        y = np.array([y, y])
    
    plt.figure(figsize=fig_size)
    
    # Generate spectrograms for each channel
    for i in range(2):
        # Compute spectrogram
        S = librosa.amplitude_to_db(np.abs(librosa.stft(y[i])), ref=np.max)
        
        plt.subplot(2, 1, i+1)
        librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Channel {i+1} Spectrogram')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
        return output_path
    else:
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return buf


def compare_images(image1_path, image2_path, threshold=0.1):
    """Compare two images and return the difference percentage.
    
    Args:
        image1_path: Path to the first image
        image2_path: Path to the second image
        threshold: Threshold for considering images as different (0.0-1.0)
        
    Returns:
        Tuple of (match_percentage, is_similar)
    """
    # Open images
    img1 = Image.open(image1_path).convert('RGB')
    img2 = Image.open(image2_path).convert('RGB')
    
    # Ensure same size for comparison
    if img1.size != img2.size:
        img2 = img2.resize(img1.size)
    
    # Convert to numpy arrays
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    
    # Calculate difference
    diff = np.abs(arr1.astype(float) - arr2.astype(float))
    
    # Normalize difference
    max_diff = 255.0 * 3  # Maximum possible difference per pixel (RGB)
    diff_percentage = np.sum(diff) / (arr1.size * max_diff)
    
    # Determine if images are similar
    is_similar = diff_percentage <= threshold
    
    return (1.0 - diff_percentage, is_similar)


def generate_reference_images(input_path, output_dir=None, prefix=""):
    """Generate reference waveform and spectrogram images for an audio file.
    
    Args:
        input_path: Path to the audio file
        output_dir: Directory to save the generated images (optional)
        prefix: Prefix to add to the output image filenames
        
    Returns:
        Tuple of (waveform_path, spectrogram_path)
    """
    if output_dir is None:
        output_dir = os.path.dirname(input_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    input_filename = os.path.basename(input_path)
    name_without_ext = os.path.splitext(input_filename)[0]
    
    # Generate waveform image
    waveform_path = os.path.join(output_dir, f"{prefix}{name_without_ext}_waveform.png")
    generate_waveform_image(input_path, waveform_path)
    
    # Generate spectrogram image
    spectrogram_path = os.path.join(output_dir, f"{prefix}{name_without_ext}_spectrogram.png")
    generate_spectrogram_image(input_path, spectrogram_path)
    
    return (waveform_path, spectrogram_path) 