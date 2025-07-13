from sweep.excitation_sweep import linear_frequency_sweep
import numpy as np
from matplotlib import pyplot as plt

time_points = np.linspace(0,5,2000)

x,v = linear_frequency_sweep(time_points,3,5,5,1)

plt.plot(time_points, x)
plt.plot(time_points, v)
plt.show()
