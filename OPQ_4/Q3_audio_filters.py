import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

# --- 1. Load Audio File ---
folder = 'D:\\ict\\sem5\\DSIP\\lab & theory session\\assignmentAndOEQ\\OPQ_4\\'
filename = folder + 'song.mp3'
y, sr = librosa.load(filename, sr=None)
print(f"Sample rate: {sr}, Duration: {len(y)/sr:.2f} seconds")

# --- 2. Function to display spectrogram (STFT) ---
def plot_spectrogram(signal, sr, title):
    D = librosa.amplitude_to_db(np.abs(librosa.stft(signal, hop_length=1024)), ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, sr=sr, hop_length=1024, x_axis='time', y_axis='hz', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    plt.show()

# Q3 (a) Low-pass filter with three different cutoff frequencies
def lowpass_filter(data, cutoff, fs, order=6):
    nyq = 0.5 * fs
    sos = butter(order, cutoff / nyq, btype='low', output='sos')
    return sosfiltfilt(sos, data)

plot_spectrogram(y, sr, 'Original Audio - Full Spectrum')

cutoff_list = [500, 2000, 5000]   # cutoff frequencies in Hz

for cutoff in cutoff_list:
    filtered = lowpass_filter(y, cutoff, sr)
    plot_spectrogram(filtered, sr, f'Low-pass Filtered Audio (Cutoff = {cutoff} Hz)')
    sf.write(f'lowpass_{cutoff}Hz.mp3', filtered, sr)
    print(f"Saved lowpass_{cutoff}Hz.mp3")

# Q3 (b) Band-pass / Band-stop filter for vocals and background music
def bandpass_filter(data, lowcut, highcut, fs, order=6):
    nyq = 0.5 * fs
    sos = butter(order, [lowcut / nyq, highcut / nyq], btype='band', output='sos')
    return sosfiltfilt(sos, data)

def bandstop_filter(data, lowcut, highcut, fs, order=6):
    nyq = 0.5 * fs
    sos = butter(order, [lowcut / nyq, highcut / nyq], btype='bandstop', output='sos')
    return sosfiltfilt(sos, data)

# Vocal range is taken as 300 - 3400 Hz
# Band-pass keeps vocals and suppresses background music
vocals_only = bandpass_filter(y, 300, 3400, sr)

# Band-stop removes vocals and keeps background music
background_only = bandstop_filter(y, 300, 3400, sr)

plot_spectrogram(vocals_only, sr, 'Background Music Suppressed (Band-pass 300-3400 Hz)')
plot_spectrogram(background_only, sr, 'Vocals Suppressed (Band-stop 300-3400 Hz)')

# --- Save filtered audio as MP3 ---
sf.write('vocals_only.mp3', vocals_only, sr)
sf.write('background_only.mp3', background_only, sr)
print("Saved vocals_only.mp3 and background_only.mp3")
