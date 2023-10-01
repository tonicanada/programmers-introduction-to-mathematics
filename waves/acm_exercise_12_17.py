import waves
import numpy as np
import math
import sympy as sp


def exercise_12_17():
    y_initial = np.array([0, 0.5, 0, 0, 0])
    A = waves.bead_matrix()
    eigenvalues, eigenvectors = np.linalg.eig(A)
    sorted_indices = np.argsort(eigenvalues)[::-1]  
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    z_initial = eigenvectors.dot(y_initial)
    t = sp.symbols('t')
    z = [0] * len(A)

    for i in range(len(A)):
        z[i] = z_initial[i] * sp.cos(sp.sqrt(-eigenvalues[i])*t)

    # Convertimos a base original
    y = eigenvectors.dot(z)
    print(y)

exercise_12_17()