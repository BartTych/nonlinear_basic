from sweep.excitation_sweep import linear_frequency_sweep
import numpy as np
from scipy.sparse import lil_matrix
from matplotlib import pyplot as plt
from scipy.signal import hilbert


def calculate_system_with_a(l_multi, n_multiplier,n_exp,amplitude):

    n_springs = 1
    n_nodes = 2

    # this should be separate matrix assembly method
    C = lil_matrix((n_springs, n_nodes))

    C[0, 0] = -1  # Spring 0: 0 ↔ 1
    C[0, 1] = +1

    C = C.tocsr()

    k = np.array([10000.0])      # stiffnesses
    a = np.array([n_exp])           # power exponent: 2.0 means nonlinear

    u = np.array([0.0, 0.0])
    M = np.eye(2)
    M_inv = np.linalg.inv(M)

    v = np.zeros_like(u)
    acc = np.zeros_like(u)
    damping = 30.0
    u_log = []

    dt = 1e-5
    steps = int(0.7 * 10e5)

    time_steps = np.linspace(0, dt * steps, steps)
    excitation_x, excitation_v = linear_frequency_sweep(time_steps, 15, 30, dt * steps, amplitude)
    for i, time in enumerate(time_steps):

        u[0] = excitation_x[i]
        v[0] = excitation_v[i]
        #liczenie sil jest inne bo dochodzi komponenet nieliniowy
        
        du = C @ u  # spring elongations: shape (3,)
        
        f_springs = k * np.sign(du) * (l_multi * np.abs(du) + n_multiplier * np.abs(du) ** a)
        f_nodes = C.T @ f_springs
        
        acc = M_inv @ (-f_nodes - v * damping)
        # semi implicit
        u += v * dt
        v += acc * dt
        # explicit
        u += v * dt
        
        if i % 100 == 0:
            u_log.append(u.copy())

    u_array = np.array(u_log)
    return u_array

#u_1 = calculate_system_with_a(1, 0.0001, 2.0)

for n in np.linspace(0.0005, 0.010, 7):
    u_1 = calculate_system_with_a(0.8, 5.2, 2.0, n)
    analytic_signal = hilbert(u_1[:, 1])
    amplitude_envelope = np.abs(analytic_signal)
    plt.plot(amplitude_envelope)

#u_2 = calculate_system_with_a(0.8, 3.2, 2.0, 0.006)
#u_3 = calculate_system_with_a(0.8, 3.2, 2.0, 0.007)
#u_4 = calculate_system_with_a(0.8, 3.2, 2.0, 0.008)

#plt.plot(u_1[:, 0])

#plt.plot(amplitude_envelope)
#plt.plot(u_2[:, 1])
#plt.plot(u_3[:, 1])
#plt.plot(u_4[:, 1])


plt.show()
