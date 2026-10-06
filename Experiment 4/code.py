import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, cheby1, lfilter, freqz, bilinear

# ==========================================
# Experiment 4: Filter Design & Filtering
# ==========================================

def design_butterworth_filter(filter_order, cutoff_frequency, sampling_frequency):
    analog_b, analog_a = butter(filter_order, cutoff_frequency, analog=True, btype='low')
    digital_b, digital_a = bilinear(analog_b, analog_a, sampling_frequency)
    return digital_b, digital_a

def design_chebyshev_filter(filter_order, cutoff_frequency, sampling_frequency, ripple):
    analog_b, analog_a = cheby1(filter_order, ripple, cutoff_frequency, analog=True, btype='low')
    digital_b, digital_a = bilinear(analog_b, analog_a, sampling_frequency)
    return digital_b, digital_a

def plot_filter_response(digital_b, digital_a, sampling_frequency):
    frequency, magnitude_response = freqz(digital_b, digital_a, fs=sampling_frequency)
    
    plt.figure(figsize=(10, 6))
    plt.plot(frequency, np.abs(magnitude_response))
    plt.title('Filter Magnitude Response')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.show()

    _, impulse_response = freqz(digital_b, digital_a, fs=sampling_frequency, worN=4096)
    
    plt.figure(figsize=(10, 6))
    plt.plot(impulse_response)
    plt.title('Filter Impulse Response')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

filter_order = 4
cutoff_frequency = 1000
sampling_frequency = 8000
ripple = 0.5

digital_b, digital_a = design_butterworth_filter(filter_order, cutoff_frequency, sampling_frequency)
plot_filter_response(digital_b, digital_a, sampling_frequency)

digital_b, digital_a = design_chebyshev_filter(filter_order, cutoff_frequency, sampling_frequency, ripple)
plot_filter_response(digital_b, digital_a, sampling_frequency)

filter_path = 'filter_coefficients.txt'
np.savetxt(filter_path, np.vstack((digital_b, digital_a)), delimiter=',')
print(f"Filter coefficients saved at: {filter_path}")


# ==========================================
# Additional Task
# ==========================================

fs = 10000
duration = 0.005
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

f1, f2, f3, f4 = 1000, 2000, 3000, 4000

s1 = np.sin(2 * np.pi * f1 * t)
s2 = np.sin(2 * np.pi * f2 * t)
s3 = np.sin(2 * np.pi * f3 * t)
s4 = np.sin(2 * np.pi * f4 * t)

plt.figure(figsize=(10, 8))
plt.subplot(4, 1, 1)
plt.plot(t * 1000, s1)
plt.title('1 kHz Sine Wave')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(t * 1000, s2)
plt.title('2 kHz Sine Wave')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(t * 1000, s3)
plt.title('3 kHz Sine Wave')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(t * 1000, s4)
plt.title('4 kHz Sine Wave')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.tight_layout()
plt.show()

N = 4000
t_fft = np.linspace(0, N / fs, N, endpoint=False)
s_composite = (np.sin(2 * np.pi * f1 * t_fft) + 
               np.sin(2 * np.pi * f2 * t_fft) + 
               np.sin(2 * np.pi * f3 * t_fft) + 
               np.sin(2 * np.pi * f4 * t_fft))

freqs = np.fft.fftfreq(N, 1/fs)[:N//2]
fft_composite = np.abs(np.fft.fft(s_composite)[:N//2]) * (2 / N)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(t_fft[:int(fs*0.005)] * 1000, s_composite[:int(fs*0.005)])
plt.title('Composite Signal (Time Domain)')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(freqs, fft_composite)
plt.title('Composite Signal (Frequency Domain)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.tight_layout()
plt.show()

digital_b_bw2k, digital_a_bw2k = design_butterworth_filter(filter_order=4, cutoff_frequency=2000, sampling_frequency=fs)
filtered_bw = lfilter(digital_b_bw2k, digital_a_bw2k, s_composite)
fft_bw = np.abs(np.fft.fft(filtered_bw)[:N//2]) * (2 / N)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(t_fft[:int(fs*0.005)] * 1000, filtered_bw[:int(fs*0.005)])
plt.title('Butterworth Filter Output (Time Domain)')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(freqs, fft_bw)
plt.title('Butterworth Filter Output (Frequency Domain)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.tight_layout()
plt.show()

digital_b_ch3k, digital_a_ch3k = design_chebyshev_filter(filter_order=4, cutoff_frequency=3000, sampling_frequency=fs, ripple=0.5)
filtered_ch = lfilter(digital_b_ch3k, digital_a_ch3k, s_composite)
fft_ch = np.abs(np.fft.fft(filtered_ch)[:N//2]) * (2 / N)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(t_fft[:int(fs*0.005)] * 1000, filtered_ch[:int(fs*0.005)])
plt.title('Chebyshev Filter Output (Time Domain)')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(freqs, fft_ch)
plt.title('Chebyshev Filter Output (Frequency Domain)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.tight_layout()
plt.show()