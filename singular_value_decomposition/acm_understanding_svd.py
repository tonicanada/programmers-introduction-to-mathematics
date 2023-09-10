import sympy as sp
import numpy as np
from scipy.optimize import fsolve
from svd import svd_1d


def generar_puntos(m, n, coeficientes=None, ruido=0.1, seed=42):
    """
    Genera m puntos aleatorios en n dimensiones.

    Parámetros:
    m (int): Número de puntos a generar.
    n (int): Número de dimensiones.
    coeficientes (list, opcional): Lista de coeficientes para las dimensiones. Si no se proporciona, se generan aleatoriamente.
    ruido (float, opcional): Magnitud del ruido a añadir a los puntos. Por defecto es 0.1.
    seed (int, opcional): Semilla para la generación de números aleatorios. Por defecto es 42.

    Retorna:
    list: Lista de m puntos (cada punto es una lista) en n dimensiones.
    """
    np.random.seed(seed)

    # Si no se proporcionan coeficientes, generarlos aleatoriamente
    if coeficientes is None:
        coeficientes = np.random.rand(n)

    # Generar valores aleatorios para la primera dimensión
    x = np.random.rand(m)

    # Generar puntos en n dimensiones
    puntos = []
    for i in range(m):
        punto = [x[i]]
        for j in range(1, n):
            valor = coeficientes[j] * x[i] + ruido * np.random.randn()
            punto.append(valor)
        puntos.append(punto)

    return puntos


def reduce_rango_matrix(A, l):
    for i in range(l):
        A[i] = 0
    return A


def find_v_which_maximize_projection(m, n):
    # Creamos el vector v que queremos encontrar
    v = []
    for i in range(1, n + 1):
        v.append(sp.Symbol(f'v{i}'))
    v = sp.Matrix(v)

    # Generamos vector Av
    av = []
    w = generar_puntos(m, n)

    for i in range(len(w)):
        w[i] = sp.Matrix(w[i])
        av.append(w[i].dot(v))
    av = sp.Matrix(av)
    av_squared = av.dot(av)

    # Generamos restricción |v|=1
    restriction_mod_v = 0
    for i in range(len(v)):
        restriction_mod_v += v[i]**2
    restriction_mod_v += -1
    print(restriction_mod_v)

    # Generamos expresion con multiplicador de Lagrange
    lmbda = sp.Symbol('lambda')
    L = av_squared + lmbda * (restriction_mod_v)

    # Generamos sistema de ecuaciones
    eq_system = []
    for i in range(len(v)):
        eq_system.append(sp.Eq(L.diff(v[i]), 0))
    eq_system.append(sp.Eq(L.diff(lmbda), 0))
    print(eq_system)
    sol = sp.solve(eq_system)

    return sol


v1 = find_v_which_maximize_projection(30, 2)

v1_prime = svd_1d(np.array(generar_puntos(30, 2)))

print(v1)
print(v1_prime)