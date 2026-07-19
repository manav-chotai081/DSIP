import numpy as np
import matplotlib.pyplot as plt
def simulate_function(time):
    y = np.zeros_like(time)
    y[time >= 0] = 1
    y[time >= 3] -= 1
    y[time >= 7] -= 5
    return y
# Define the time range
time = np.linspace(-10, 10, 1000)
# Simulate the function
function_values = simulate_function(time)
# Plot and display the function
plt.plot(time, function_values)
plt.title('Function y(t) = u(t) - u(t-3) - 5*u(t-7)')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.ylim([-5.5, 1.5])
plt.grid(True)
plt.show()