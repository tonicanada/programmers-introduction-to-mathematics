import sympy as sp
import numpy as np


# 12.20. Consider a one-dimensional “bead system” where instead of the beads physically
# moving, they are given some initial heat. Adjacent beads transfer heat between them
# according to a discrete version of the so-called heat equation. Find an exposition of the
# discrete heat equation online that allows you to set up a linear system and solve it for 10
# beads. What do the eigenvalues of this system look like?

def exercise_12_20(n, r_val):
    A = sp.zeros(n)
    r = sp.symbols('r')
    for i in range(n):
        for j in range(i, n):
            if i == j:
                A[i, j] = 1 - 2 * r
            elif (j == i + 1) and (i != n):
                A[i, j] = r
            A[j, i] = A[i, j]

    A_np = sp.lambdify(r, A, 'numpy')
    eigenvalues, eigenvectors = np.linalg.eig(A_np(r_val))

    print("eigenvectors:", eigenvectors)
    print("eigenvalues:", eigenvalues)


exercise_12_20(10,3)
