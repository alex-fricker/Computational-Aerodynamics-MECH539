import numpy as np
import matplotlib.pyplot as plt
from quasi_1D_euler import quasi1DEuler

n_grid = 50
pressure_ratio = 0.8
epsilon = 0.5
max_itr = 1000

euler_solver = quasi1DEuler(n=n_grid, epsilon=epsilon, max_iterations=max_itr)
residual = euler_solver.solve(exit_pressure_ratio=pressure_ratio)

fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax1.plot(np.arange(1, len(residual) + 1), residual)

print("Done")
