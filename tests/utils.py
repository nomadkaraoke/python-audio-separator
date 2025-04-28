import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import soundfile as sf
from pathlib import Path
from skimage.metrics import structural_similarity as ssim


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
    
    # Plot waveform for each channel with fixed Y-axis scale
    plt.subplot(2, 1, 1)
    plt.plot(y[0])
    plt.title('Channel 1')
    plt.ylim([-1.0, 1.0])  # Fixed Y-axis scale for all waveforms
    
    plt.subplot(2, 1, 2)
    plt.plot(y[1])
    plt.title('Channel 2')
    plt.ylim([-1.0, 1.0])  # Fixed Y-axis scale for all waveforms
    
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
    
    # Set fixed min and max values for spectrogram color scale
    vmin = -80  # dB
    vmax = 0    # dB
    
    # Generate spectrograms for each channel
    for i in range(2):
        # Compute spectrogram
        S = librosa.amplitude_to_db(np.abs(librosa.stft(y[i])), ref=np.max)
        
        plt.subplot(2, 1, i+1)
        # Use fixed frequency range and consistent color scaling
        librosa.display.specshow(
            S, 
            sr=sr, 
            x_axis='time', 
            y_axis='log',
            vmin=vmin,
            vmax=vmax
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Channel {i+1} Spectrogram')
        
        # Set frequency range (y-axis) - typically up to Nyquist frequency (sr/2)
        plt.ylim([20, sr/2])  # From 20Hz to Nyquist frequency
    
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


def compare_images(image1_path, image2_path, min_similarity_threshold=0.999):
    """Compare two images using Structural Similarity Index (SSIM) which is robust to small shifts.
    
    Args:
        image1_path: Path to the first image
        image2_path: Path to the second image
        min_similarity_threshold: Minimum similarity required for images to be considered matching (0.0-1.0)
            - Higher values (closer to 1.0) require images to be more similar
            - Lower values (closer to 0.0) are more permissive
            - A value of 0.99 requires 99% similarity between images
            - A value of 0.0 would consider any images to match
        
    Returns:
        Tuple of (similarity_score, is_match)
        - similarity_score: Value between 0.0 and 1.0, where 1.0 means identical images
        - is_match: Boolean indicating if similarity_score >= min_similarity_threshold
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
    
    # Calculate SSIM for each color channel
    similarity_scores = []
    for channel in range(3):  # RGB channels
        score = ssim(arr1[:,:,channel], arr2[:,:,channel], data_range=255)
        similarity_scores.append(score)
    
    # Calculate average SSIM across channels
    similarity_score = np.mean(similarity_scores)
    
    # Determine if images match by comparing similarity to threshold
    is_match = similarity_score >= min_similarity_threshold
    
    return (similarity_score, is_match)


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