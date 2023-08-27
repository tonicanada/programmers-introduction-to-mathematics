import sympy as sp
import random
from tabulate import tabulate
import numpy as np
import math


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


def div_diff_recurrent(k, j, points, memo):
    """
    Calculate the divided differences for a set of points using a recursive approach.

    Parameters:
    - k (int): The starting index of the current divided difference.
    - j (int): The ending index of the current divided difference.
    - points (list of lists): A list containing the x, y values of the points. 
                               Each list is of the form [x, y].
    - memo (dict): A dictionary used for memoization to store already computed divided differences.

    Returns:
    - float or symbolic expression (depends on the points input form): The calculated divided difference for the range [k, j].

    Note:
    - The function uses memoization to avoid redundant calculations.
    """
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
    """
    Compute the divided differences for a given set of points.

    Parameters:
    - points (list of lists): A list containing the x, y values of the points. 
                               Each tuple is of the form [x, y].

    Returns:
    - dict: A dictionary containing the computed divided differences. The keys represent 
            the range [k, j] and the values are the corresponding divided differences.
    """
    n = len(points)
    divided_diff_dict = {}
    div_diff_recurrent(0, n, points, divided_diff_dict)
    return divided_diff_dict


def get_coef_lagrangepolynomial_basis(points, x):
    n = len(points)
    coef = []
    for i in range(n):
        term = 1
        for j in range(n):
            if i != j:
                term = term * ((x - points[j][0]) / (points[i][0] - points[j][0]))
        coef.append(sp.Poly(term, x).all_coeffs()[::-1])
    y = sp.Matrix([y for x, y in points])
    return coef, y


def get_coef_newtonpolynomial_basis(points, x):
    n = len(points)
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

    y = sp.Matrix(y)

    return coef, y


def get_matrix_form_interpolation_polynomial(n, numeric_points=False, polynomial='lagrange'):
    if numeric_points:
        points = generate_points(n)
        print(points)
    else:
        points = generate_points_symbolic(n)

    x = sp.symbols('x')
    if polynomial == 'lagrange':
        coef, y = get_coef_lagrangepolynomial_basis(points, x)
    elif polynomial == 'newton':
        coef, y = get_coef_newtonpolynomial_basis(points, x)

    for i in range(len(coef)):
        for j in range(len(coef[i])):
            coef[i][j] = sp.factor(coef[i][j])

    A = sp.Matrix(coef).T
    print(tabulate(coef, 'grid'))

    terms_standard_base = A * y
    coef_standard_base = [elem for elem in terms_standard_base]
    poly = sp.Poly(sum(coef * x**i for i, coef in enumerate((coef_standard_base))), x)

    poly = poly.simplify()
    print("CHECK", poly.subs(x, points[1][0]).simplify())
    return poly, A


# Ejercicio 10.13
# The Bernstein basis is a basis of the vector space of polynomials of degree at most
# n. In an exercise from Chapter 2, you explored this basis in terms of Bézier curves. Like
# Taylor polynomials, Bernstein polynomials can be used to approximate functions R → R
# to arbitrary accuracy. Look up the definition of the Bernstein basis, and read a theorem
# that proves they can be used to approximate functions arbitrarily well.

def get_linearinterpolation_between_points(p1, p2, t):
    lerp = p1 + (p2 - p1) * t
    return lerp


def get_bezier_curve(n):
    p = generate_points(n, seed=50)
    # p = generate_points_symbolic(3)

    p = np.array(p)

    lerp = p.copy()
    t = sp.Symbol('t')

    while len(lerp) > 1:
        m = len(lerp)
        for i in range(len(lerp) - 1):
            lerp = np.vstack(
                (lerp, get_linearinterpolation_between_points(lerp[i], lerp[i + 1], t)))

        lerp = lerp[m:]

    lerp[0][0] = lerp[0][0].simplify()
    lerp[0][1] = lerp[0][1].simplify()

    print(lerp[0].tolist())

    return lerp[0].tolist()


def get_bezier_curve_in_terms_of_points(n):
    p = []
    for i in range(n):
        p.append(sp.symbols(f'p{i}'))
    t = sp.Symbol('t')

    lerp = p.copy()

    while len(lerp) > 1:
        m = len(lerp)
        for i in range(len(lerp) - 1):
            lerp.append(get_linearinterpolation_between_points(lerp[i], lerp[i + 1], t))
        lerp = lerp[m:]

    lerp[0] = lerp[0].simplify()
    lerp[0] = sp.collect(lerp[0], t)

    print(lerp[0])

    return lerp[0]


def get_matrix_form_bernstein_polynomial(n):
    p = []
    for i in range(n+1):
        p.append(sp.symbols(f'p{i}'))
    t = sp.Symbol('t')

    coef = []
    for i in range(n+1):
        print(i, n)
        bernstein_in = math.comb(n, i) * (1-t)**(n-i) * t**i
        coef.append(sp.Poly(bernstein_in, t).all_coeffs()[::-1])

    A = sp.Matrix(coef).T
    print(tabulate(coef, 'grid'))

    y = sp.Matrix(p)

    terms_standard_base = A * y

    print(terms_standard_base)

    print(A.inv()*terms_standard_base)
    # coef_standard_base = [elem for elem in terms_standard_base]
    # poly = sp.Poly(sum(coef * t**i for i, coef in enumerate((coef_standard_base))), t)

    # poly = poly.simplify()
    # print("CHECK", poly.subs(t, p[1]).simplify())
    # return poly, A








# get_matrix_form_interpolation_polynomial(4, numeric_points=True, polynomial='lagrange')
# get_matrix_form_lagrange_polynomial(4, numeric_points=True)

# get_bezier_curve(3)
get_bezier_curve_in_terms_of_points(5)
get_matrix_form_bernstein_polynomial(4)