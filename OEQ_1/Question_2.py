import numpy as np
import matplotlib.pyplot as plt
def simulate_function(time):
    y = np.zeros_like(time)
    y[time >= 0] = 1
    y[time >= 1] += 1
    y[time >= 2] += 1
    y[time >= 3] -= 1
    y[time >= 4] -= 1
    y[time >= 5] -= 1
    return y
# Define the time range
time = np.linspace(-10, 10, 1000)
# Simulate the function
function_values = simulate_function(time)
# Plot and display the function
plt.plot(time, function_values)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.ylim([-0.5, 5.5])
plt.grid(True)
plt.show()