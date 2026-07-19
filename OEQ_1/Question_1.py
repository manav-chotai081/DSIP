import numpy as np
import matplotlib.pyplot as plt
def simulate_impulse_train(n_range, period):
    impulse_train = np.zeros(len(n_range))
    for idx, n in enumerate(n_range):
        if n % period == 0:
            impulse_train[idx] = 2
    return impulse_train
def simulate_impulse_train1(n_range1, period1):
    impulse_train = np.zeros(len(n_range1))
    for idx, n in enumerate(n_range1):
        if n % period1 != 0:
            impulse_train[idx] = 3
    return impulse_train
# Define the parameters for the impulse train
n_range1 = np.arange(-9, 11)
period1 = 2 # Period of the impulse train
# Simulate the impulse train
impulse_train1 = simulate_impulse_train1(n_range1, period1)
# Plot and display the impulse train

# Define the parameters for the impulse train
n_range = np.arange(-10, 11)
period = 2 # Period of the impulse train
# Simulate the impulse train
impulse_train = simulate_impulse_train(n_range, period)
# Plot and display the impulse train
plt.stem(n_range, impulse_train, linefmt='b-', markerfmt='bo')
plt.stem(n_range1, impulse_train1, linefmt='b-', markerfmt='bo')
plt.title('Impulse Train')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.show()