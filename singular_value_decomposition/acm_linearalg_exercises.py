import sympy as sp
import random
from tabulate import tabulate
import json


# Ejercicio 10.10
# Prove that the set of all Fibonacci-type sequences form a vector space (under what operations?).
# Find a basis, and thus compute its dimension.


def get_fibonacci_element(sym_a, sym_b, i, memo):
    if i == 0:
        memo[0] = sym_a
        return sym_a
    elif i == 1:
        memo[1] = sym_b
        return sym_b
    elif memo[i]:
        return memo[i]
    else:
        ans = get_fibonacci_element(sym_a, sym_b, i - 1, memo) + \
            get_fibonacci_element(sym_a, sym_b, i - 2, memo)
        memo[i] = ans
        return ans


def get_fibonacci_sequence(n=20):
    a = sp.symbols('a')
    b = sp.symbols('b')
    memo = [None] * (n + 1)
    get_fibonacci_element(a, b, n, memo)
    print(memo)
    return memo


# Ejercicio 10.11
# In Chapter 2 we defined and derived an algorithm for polynomial interpolation.
# Reminder: given a set of n + 1 points (x0 , y0 ), . . . , (xn , yn ), with no two xi the same,
# there is a unique degree-at-most-n polynomial passing through those points. Rephrase
# this problem as solving a matrix-vector multiplication problem Ay = x for y. Hint: A
# should be an (n + 1) × (n + 1) matrix.

def generate_points(n, range_min=-10, range_max=10, seed=42):
    """
    Generates n points with random integer coordinates ensuring unique x-values.

    Parameters:
    - n: Number of points to generate.
    - range_min: Minimum value for the coordinates.
    - range_max: Maximum value for the coordinates.
    - seed: Optional seed for reproducibility.

    Returns:
    - List of n points with random integer coordinates.
    """
    if seed is not None:
        random.seed(seed)

    points = []
    used_x = set()  # Set to store used x-values

    for _ in range(n):
        x = random.randint(range_min, range_max)
        
        # Keep generating new x-values until we find one that hasn't been used
        while x in used_x:
            x = random.randint(range_min, range_max)
        
        used_x.add(x)
        y = random.randint(range_min, range_max)
        points.append([x, y])

    return points


def generate_points_symbolic(n):
    points_dict = {}
    for i in range(1, n + 1):
        points_dict[i] = [sp.symbols(f'x{i}'), sp.symbols(f'y{i}')]
    points = list(points_dict.values())
    return points


def get_coef_lagrangepolynomial_basis(points):
    n = len(points)
    x = sp.symbols('x')
    coef = []
    for i in range(n):
        term = 1
        for j in range(n):
            if i != j:
                term = term * ((x - points[j][0]) / (points[i][0] - points[j][0]))
        coef.append(sp.Poly(term, x).all_coeffs()[::-1])
    return coef


def div_diff_recurrent(k, j, points, memo):
    if f"{k+1}-{j}" in memo:
        return memo[f"{k+1}-{j}"]

    # Base case
    n = len(points)
    if j - k == 1:
        memo[f"{k+1}-{j}"] = points[0][1]
        return points[0][1]

    a = (div_diff_recurrent(k + 1, j, points[1:n], memo) -
         div_diff_recurrent(k, j - 1, points[:n - 1], memo)) / (points[n - 1][0] - points[0][0])
    memo[f"{k+1}-{j}"] = a
    return a


def get_divided_differences(points):
    n = len(points)
    divided_diff_dict = {}
    div_diff_recurrent(0, n, points, divided_diff_dict)
    return divided_diff_dict


def get_matrix_form_newton_polynomial(n, numeric_points=False):
    if numeric_points:
        points = generate_points(n)
        print(points)
    else:
        points = generate_points_symbolic(n)

    n = len(points)
    x = sp.symbols('x')
    coef = []
    term = 1
    for i in range(n):
        coef.append(sp.Poly(term, x).all_coeffs()[::-1])
        term = term * (x - points[i][0])

    coef_same_dimension = []

    for c in coef:
        while len(c) < n:
            c.insert(n, 0)
        coef_same_dimension.append(c)

    coef = coef_same_dimension

    div_differences = get_divided_differences(points)
    y = []

    for i in range(1, n + 1):
        y.append(div_differences[f'1-{i}'])

    A = sp.Matrix(coef).T
    print(tabulate(coef, 'grid'))

    y = sp.Matrix(y)

    terms_standard_base = A * y

    coef_standard_base = [elem for elem in terms_standard_base]

    x = sp.symbols('x')
    poly = sp.Poly(sum(coef * x**i for i, coef in enumerate((coef_standard_base))), x)

    poly = poly.simplify()
    print("CHECK", poly.subs(x, points[1][0]).simplify())
    return poly, A


def get_matrix_form_lagrange_polynomial(n, numeric_points=False):
    if numeric_points:
        points = generate_points(n)
        print(points)
    else:
        points = generate_points_symbolic(n)

    coef = get_coef_lagrangepolynomial_basis(points)
    for i in range(len(coef)):
        for j in range(len(coef[i])):
            coef[i][j] = sp.factor(coef[i][j])

    A = sp.Matrix(coef).T
    print(tabulate(coef, 'grid'))

    y = sp.Matrix([y for x, y in points])
    terms_standard_base = A * y
    coef_standard_base = [elem for elem in terms_standard_base]
    x = sp.symbols('x')
    poly = sp.Poly(sum(coef * x**i for i, coef in enumerate((coef_standard_base))), x)

    poly = poly.simplify()
    print("CHECK", poly.subs(x, points[1][0]).simplify())
    return poly, A


get_matrix_form_newton_polynomial(4, numeric_points=True)
get_matrix_form_lagrange_polynomial(4, numeric_points=True)
