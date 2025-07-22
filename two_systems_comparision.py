import pickle
from matplotlib import pyplot as plt
from scipy.signal import hilbert
import numpy as np

one_spring = pickle.load(open('one_spring.pkl', 'rb'))
two_springs = pickle.load(open('two_springs.pkl', 'rb'))


for n in one_spring:
    analytic_signal = hilbert(n[:, 1])
    amplitude_envelope = np.abs(analytic_signal)
    plt.plot(amplitude_envelope)

for n in two_springs:
    analytic_signal = hilbert(n[:, 1])
    amplitude_envelope = np.abs(analytic_signal)
    plt.plot(amplitude_envelope)

plt.show()


