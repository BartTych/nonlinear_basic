import numpy as np
from scipy.sparse import lil_matrix
from matplotlib import pyplot as plt
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

k = np.array([100.0, 50.0, 100.0])      # stiffnesses
a = np.array([1.0, 1.05, 1.0])           # power exponent: 2.0 means nonlinear

u = np.array([-0.01, -0.005, 0.005, 0.01])
M = np.eye(4)
M_inv = np.linalg.inv(M)

v = np.zeros_like(u)
acc = np.zeros_like(u)
u_log = []

dt = 0.000001
steps = int(3*10e4)

for step in range(steps):
    #sily to jest inne bo dochodzi komponenet nieliniowy
      
    du = C @ u  # spring elongations: shape (3,)
    
    f_springs = k * np.sign(du) * np.abs(du) ** a
    f_nodes = C.T @ f_springs 
    
    acc = M_inv @ -f_nodes
    
    v += acc * dt

    #przemieszczenia
    u += v*dt

    if step % 10==0:
        u_log.append(u.copy())

u_array = np.array(u_log)

k = np.array([100.0, 50.0, 100.0])      # stiffnesses
a = np.array([1.0, 1.0, 1.0])           # power exponent: 2.0 means nonlinear

u = np.array([-0.01, -0.005, 0.005, 0.01])
M = np.eye(4)
M_inv = np.linalg.inv(M)

v = np.zeros_like(u)
acc = np.zeros_like(u)
u_log_2 = []

dt = 0.000001
steps = int(3*10e4)

for step in range(steps):
    #sily to jest inne bo dochodzi komponenet nieliniowy
      
    du = C @ u  # spring elongations: shape (3,)
    
    f_springs = k * np.sign(du) * np.abs(du) ** a
    f_nodes = C.T @ f_springs 
    
    acc = M_inv @ -f_nodes
    
    v += acc * dt

    #przemieszczenia
    u += v * dt

    if step % 10==0:
        u_log_2.append(u.copy())

u_array_2 = np.array(u_log_2)

plt.plot(u_array[:, 0])
plt.plot(u_array[:, 1])
plt.plot(u_array[:, 2])
plt.plot(u_array[:, 3])


plt.plot(u_array_2[:, 0])
plt.plot(u_array_2[:, 1])
plt.plot(u_array_2[:, 2])
plt.plot(u_array_2[:, 3])


plt.show()
