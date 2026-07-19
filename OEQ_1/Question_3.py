import numpy as np
import matplotlib.pyplot as plt
def simulate_continuous_ramp(time, slope):
    ramp = np.zeros_like(time)
    ramp[(time >= -2) & (time < -1)] = -2
    ramp[(time >= -1) & (time <= 1)] = slope * time[(time >= -1) & (time <= 1)]
    ramp[(time > 1) & (time <= 2)] = 2
    return ramp
def simulate_discrete_ramp(num_samples, slope):
    # Not used in this version since the reference image is continuous, but kept for structure
    return np.zeros(num_samples)
# Define the time range for the continuous ramp signal
time = np.linspace(-3.5, 3.5, 1000) # Time range from -5 to 5
# Define the number of samples and slope for the discrete ramp signal
num_samples = 20 # Number of samples
slope = 2 # Slope of the ramp
# Simulate the continuous ramp signal
continuous_ramp = simulate_continuous_ramp(time, slope)
# Simulate the discrete ramp signal
discrete_ramp = simulate_discrete_ramp(num_samples, slope)
# Plot and display the continuous and discrete ramp signals
ax = plt.gca()
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.plot(time, continuous_ramp, 'b-')
plt.title('Continuous Ramp Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.xticks([-2, -1, 1, 2])
plt.yticks([-2, 2])
plt.show()