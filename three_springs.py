from sweep.excitation_sweep import linear_frequency_sweep
import numpy as np
from scipy.sparse import lil_matrix
from matplotlib import pyplot as plt
from scipy.signal import hilbert
import pickle

def calculate_system_with_a(l_multi, n_multiplier,n_exp,amplitude):

    n_springs = 3
    n_nodes = 4

    # this should be separate matrix assembly method
    C = lil_matrix((n_springs, n_nodes))

    C[0, 0] = -1  # Spring 0: 0 ↔ 1
    C[0, 1] = +1

    C[1, 1] = -1  # Spring 1: 1 ↔ 2 (nonlinear)
    C[1, 2] = +1

    C[2, 2] = -1  # Spring 2: 2 ↔ 3
    C[2, 3] = +1

    C = C.tocsr()

    k = np.array([40000.0, 40000.0, 40000.0])      # stiffnesses
    a = np.array([n_exp, n_exp, n_exp])           # power exponent: 2.0 means nonlinear

    u = np.array([0.0, 0.0, 0.0, 0.0])
    M = np.eye(4)
    M_inv = np.linalg.inv(M)

    v = np.zeros_like(u)
    acc = np.zeros_like(u)
    damping = 30.0
    u_log = []

    dt = 1e-5
    steps = int(1.3 * 10e5)

    time_steps = np.linspace(0, dt * steps, steps)
    excitation_x, excitation_v = linear_frequency_sweep(time_steps, 5, 120, dt * steps, amplitude)
    
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
data_storage = []
for n in np.linspace(0, 15, 5):
    u_1 = calculate_system_with_a(0.8, n, 2.0, 0.01)
    data_storage.append(u_1)    
    analytic_signal = hilbert(u_1[:, 3])
    amplitude_envelope = np.abs(analytic_signal)
    plt.plot(amplitude_envelope)

pickle.dump((data_storage), open(f'two_springs.pkl', 'wb'))
#u_2 = calculate_system_with_a(0.8, 3.2, 2.0, 0.006)
#u_3 = calculate_system_with_a(0.8, 3.2, 2.0, 0.007)
#u_4 = calculate_system_with_a(0.8, 3.2, 2.0, 0.008)

#plt.plot(u_1[:, 0])

#plt.plot(amplitude_envelope)
#plt.plot(u_2[:, 1])
#plt.plot(u_3[:, 1])
#plt.plot(u_4[:, 1])


plt.show()
