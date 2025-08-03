import numpy as np
import matplotlib.pyplot as plt
import librosa
import scipy.signal as sig
from numba import njit

@njit
def filter_overtones_numba(frequencies, atol=0.01):
    frequencies = np.sort(frequencies)  # Sort frequencies in ascending order
    fundamentals = []
    
    for freq in frequencies:
        is_overtone = False
        for fundamental in fundamentals:
            if abs(freq / fundamental - round(freq / fundamental)) < atol:  # Check if freq is a multiple of a fundamental
                is_overtone = True
                break
        if not is_overtone:
            fundamentals.append(freq)
    
    return np.array(fundamentals)

def compute_fft(signal, sample_rate):
    fft_result = np.fft.fft(signal)  # FFT of the signal
    frequencies = np.fft.fftfreq(len(fft_result), d=1/sample_rate)  # Frequency array
    magnitude = np.abs(fft_result)  # Magnitude of FFT
    return fft_result, frequencies, magnitude

@njit
def filter_fft_by_notes(fft_result, frequencies, notes, cleanup_threshold):
    filtered_fft_result = np.zeros_like(fft_result)
    for target_frequency in notes:
        idx = np.argmin(np.abs(frequencies - target_frequency))
        filtered_fft_result[idx] = fft_result[idx]
    
    # Apply cleanup threshold
    filtered_fft_result[np.abs(filtered_fft_result) < cleanup_threshold] = 0
    return filtered_fft_result

# Sampling rate and time array
rate = 2500  # Sampling rate
interval = 1.0 / rate  # Sampling frequency
t = np.arange(0, 1, interval)  # Time array

# Generate signal
data_array_1 = 2 * np.sin(2 * np.pi * 2 * t)  # Sinusoid of frequency 2 Hz
data_array_2 = np.sin(2 * np.pi * 3 * t)  # Sinusoid of frequency 3 Hz
signal = data_array_1 + data_array_2  # Sum of sinusoids

# Fourier transform of the signal
fft_result, frequencies, magnitude = compute_fft(signal, rate)

# Generate notes
N = 100  # Number of keys
keys = np.arange(1, N, 1)
notes = (np.power(2, (1 / 12))) ** (keys - 49) * 440  # Frequencies of each note

# Load audio sample
sample_path = "sfg.mp3"  # Fetching sample sound
audio_signal, sample_rate = librosa.load(sample_path, sr=None, mono=False)  # Extracting audio signal

if audio_signal.ndim > 1:
    audio_signal = audio_signal[0, :]  # Use the first channel if stereo

# Low-pass filter
cutoff_frequency = 7000  # Cutoff frequency
nyquist = 1.0 * sample_rate  # Nyquist frequency
normal_cutoff = cutoff_frequency / nyquist  # Normalization of cutoff frequency
b, a = sig.butter(N=4, Wn=normal_cutoff, btype='low', analog=False)  # Low-pass filter
filtered_signal = sig.filtfilt(b, a, audio_signal)  # Apply filter

# FFT of filtered signal
fft_result, frequencies, magnitude = compute_fft(filtered_signal, sample_rate)

# Filter FFT by notes
cleanup_threshold = 40  # Cleanup threshold
filtered_fft_result = filter_fft_by_notes(fft_result, frequencies, notes, cleanup_threshold)

# Extract filtered frequencies
filtered_magnitude = np.abs(filtered_fft_result)  # Magnitude of selected amplitudes
filtered_frequencies = frequencies[filtered_magnitude > 0]
print(filtered_frequencies)

# Filter out overtones
filtered_frequencies = np.sort(filtered_frequencies)  # Sort frequencies in ascending order
fundamentals = filter_overtones_numba(filtered_frequencies, atol=0.01)
print(fundamentals)