import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import hilbert
from sweep.excitation_sweep import linear_frequency_sweep

def calculate_nonlinear_spring_system_2D(l_multi, n_multiplier, n_exp, amplitude):
    n_nodes = 3
    dim = 2  # 2D
    dofs = n_nodes * dim  # 6 DOFs

    # Node connectivity: spring connects (node_i, node_j)
    spring_connections = [
        (0, 1),  # spring 0
        (1, 2),  # spring 1
    ]

    # Initial rest positions of nodes (you can modify this layout)
    rest_positions = np.array([
    [0.0, 0.0],
    [1.0, 0.45],
    [2.0, 0.0],
])

    u = rest_positions.flatten()  # this initializes positions

# Then compute rest lengths from same rest_positions
    rest_lengths = [
    np.linalg.norm(rest_positions[j] - rest_positions[i])
    for i, j in spring_connections
]


    # Spring properties
    k = np.array([40000.0, 30000.0])
    a = np.array([n_exp, n_exp])

    # Dynamic state
    u = rest_positions.flatten()  # Flatten to 1D array: [x0, y0, x1, y1, x2, y2]
    v = np.zeros_like(u)
    acc = np.zeros_like(u)

    M = np.eye(dofs)
    M_inv = np.linalg.inv(M)
    damping = 30.0
    u_log = []
    time_log = []
    # Time integration
    dt = 1e-5
    steps = int(1 * 10e5)
    time_steps = np.linspace(0, dt * steps, steps)

    # Excitation: linear chirp
    excitation_x, excitation_v = linear_frequency_sweep(time_steps, 5, 120, dt * steps, amplitude)
    #u[3] += 1e-3  # Initial offset for node 1 in y direction
    for i, time in enumerate(time_steps):
        # Apply excitation to node 0 in x direction
        u[0:2] = excitation_x[i]
        v[0:2] = excitation_v[i]



        #u[4] = 2.0  # Fix node 2 position
        #u[5] = 0.0  # Fix node 2 position
        #v[4:6] = 0.0 # v_x and v_y of node 2
        
        #u[0:2] = rest_positions[0]
        #v[0:2] = 0.0

        u[4:6] = rest_positions[2]
        v[4:6] = 0.0
        
        # Compute spring forces
        f_total = np.zeros_like(u)

        for s, (i_node, j_node) in enumerate(spring_connections):
            xi = u[2*i_node:2*i_node+2]
            xj = u[2*j_node:2*j_node+2]
            d = xj - xi
            length = np.linalg.norm(d)

            if length > 1e-12:
                dir = d / length
                elongation = length - rest_lengths[s]
                abs_elong = np.abs(elongation)
                f_mag = k[s] * np.sign(elongation) * (
                    l_multi * abs_elong + n_multiplier * abs_elong**a[s]
                )
                f_vec = f_mag * dir

                # Apply force to global vector
                f_total[2*i_node:2*i_node+2] -= f_vec
                f_total[2*j_node:2*j_node+2] += f_vec

        # Integrate motion
        acc = M_inv @ (-f_total - damping * v)
        v += acc * dt
        u += v * dt

        if i % 10 == 0:
            u_log.append(u.copy())
            time_log.append(time)

    u_array = np.array(u_log).reshape(-1, n_nodes, dim)
    return u_array, time_log

data_storage = []
for n in np.linspace(1, 40, 5):
    print(f'Calculating for n = {n:.4f}')
    u_1, time_log = calculate_nonlinear_spring_system_2D(0.8, n, 2, 0.001)
    #data_storage.append(u_1)    
    #analytic_signal = hilbert(u_1[:, 2])
    #amplitude_envelope = np.abs(analytic_signal)
    plt.plot(u_1[:, 1, 0], u_1[:, 1, 1], label=f'n={n:.2f}')
    #plt.plot(time_log,u_1[:, 1, 1])
    
#pickle.dump((data_storage), open(f'two_springs.pkl', 'wb'))

plt.show()