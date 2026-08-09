import librosa
import numpy as np
import matplotlib.pyplot as plt

# Load audio files
song1, sr = librosa.load("kesariya.mp3", sr=None)
song2, sr = librosa.load("kesariya_instrumental.mp3", sr=None)
song3, sr = librosa.load("believer.mp3", sr=None)

# Take first 10 seconds
song1 = song1[:10  * sr]
song2 = song2[:10 * sr]
song3 = song3[:10 * sr]

# Make all songs same length
n = min(len(song1), len(song2), len(song3))

song1 = song1[:n]
song2 = song2[:n]
song3 = song3[:n]

song2 = np.roll(song2, 3000)

# Cross Correlation
corr12 = np.correlate(song1, song2, mode='full')
corr13 = np.correlate(song1, song3, mode='full')
corr23 = np.correlate(song2, song3, mode='full')

# Similarity values
sim12 = np.max(np.abs(corr12))
sim13 = np.max(np.abs(corr13))
sim23 = np.max(np.abs(corr23))

print("Similarity Values")
print("------------------")
print("Original vs Karaoke =", sim12)
print("Original vs Different Song =", sim13)
print("Karaoke vs Different Song =", sim23)

# Only one graph for analysis
plt.figure(figsize=(8,4))
plt.plot(corr12)
plt.title("Cross Correlation: Original vs Karaoke")
plt.xlabel("Lag")
plt.ylabel("Correlation")
plt.show()


# import librosa
# import numpy as np
# import matplotlib.pyplot as plt

# # Load audio files
# song1, sr = librosa.load("kesariya.mp3", sr=None)
# song2, sr = librosa.load("kesariya_instrumental.mp3", sr=None)
# song3, sr = librosa.load("believer.mp3", sr=None)

# # Take first 10 seconds
# song1 = song1[:10 * sr]
# song2 = song2[:10 * sr]
# song3 = song3[:10 * sr]

# # Make all songs same length
# n = min(len(song1), len(song2), len(song3))
# song1 = song1[:n]
# song2 = song2[:n]
# song3 = song3[:n]

# # Cross Correlation
# corr12 = np.correlate(song1, song2, mode='full')
# corr13 = np.correlate(song1, song3, mode='full')
# corr23 = np.correlate(song2, song3, mode='full')

# # Similarity values
# sim12 = np.max(np.abs(corr12))
# sim13 = np.max(np.abs(corr13))
# sim23 = np.max(np.abs(corr23))

# print("Similarity Values")
# print("------------------")
# print("Original vs Karaoke =", sim12)
# print("Original vs Different Song =", sim13)
# print("Karaoke vs Different Song =", sim23)

# # Cross Correlation Graphs
# plt.figure(figsize=(8,4))
# plt.plot(corr12)
# plt.title("Cross Correlation: Original vs Karaoke")
# plt.xlabel("Lag")
# plt.ylabel("Correlation")
# plt.show()

# plt.figure(figsize=(8,4))
# plt.plot(corr13)
# plt.title("Cross Correlation: Original vs Different Song")
# plt.xlabel("Lag")
# plt.ylabel("Correlation")
# plt.show()

# plt.figure(figsize=(8,4))
# plt.plot(corr23)
# plt.title("Cross Correlation: Karaoke vs Different Song")
# plt.xlabel("Lag")
# plt.ylabel("Correlation")
# plt.show()

# # Auto Correlation
# auto1 = np.correlate(song1, song1, mode='full')
# auto2 = np.correlate(song2, song2, mode='full')
# auto3 = np.correlate(song3, song3, mode='full')

# # Auto Correlation Graphs
# plt.figure(figsize=(8,4))
# plt.plot(auto1)
# plt.title("Auto Correlation: Original Song")
# plt.xlabel("Lag")
# plt.ylabel("Correlation")
# plt.show()

# plt.figure(figsize=(8,4))
# plt.plot(auto2)
# plt.title("Auto Correlation: Karaoke Song")
# plt.xlabel("Lag")
# plt.ylabel("Correlation")
# plt.show()

# plt.figure(figsize=(8,4))
# plt.plot(auto3)
# plt.title("Auto Correlation: Different Song")
# plt.xlabel("Lag")
# plt.ylabel("Correlation")
# plt.show()