from polynomial import Polynomial
import timeit


def get_polynomial_from_points_lagrange(points_list=[(1, 324), (3, 2383), (5, 6609)]):
    """
    Function that given a list of (n+1) points returns a n-degree polynomial
    that passes through all of them. Uses Lagrange Interpolation.
    """
    n = len(points_list)
    base_sum = Polynomial([])
    for i in range(n):
        base_product = Polynomial([1])
        for j in range(n):
            if i != j:
                b = points_list[i][0] - points_list[j][0]
                base_product *= Polynomial([-points_list[j][0] / b, 1 / b])
        base_sum += Polynomial([points_list[i][1]]) * base_product
    return base_sum


def get_polynomial_from_points_newton(points_list=[(1, 324), (3, 2383), (5, 6609)]):
    """
    Function that given a list of (n+1) points returns a n-degree polynomial
    that passes through all of them. Uses Newton Interpolation.
    """
    n = len(points_list)
    a = [None] * n

    def get_pk_newton(k, memo):
        # Base case
        if k == 0:
            return Polynomial([points_list[0][1]])
        c = 1
        # Check if result is in memo
        if memo[k] != None:
            return memo[k]

        p = Polynomial([1])
        for i in range(k):
            c *= points_list[k][0] - points_list[i][0]
            p *= Polynomial([-points_list[i][0], 1])
        res = (
            get_pk_newton(k - 1, memo)
            + Polynomial(
                [
                    (points_list[k][1] - get_pk_newton(k - 1, memo)(points_list[k][0]))
                    * (1 / c)
                ]
            )
            * p
        )
        memo[k] = res
        return res

    return get_pk_newton(n - 1, a)


def horners_method(p, x):
    """
    Function that evaluates a polynomial p on x, p(x)
    using Horner's method
    """
    reversed_coef = list(reversed(p.coefficients))
    n = len(reversed_coef)
    s = 0
    for i in range(n):
        s = s * x + reversed_coef[i]
    return s


def normalevaluation(p, x):
    """
    Function that evaluates a polynomial p on x, p(x)
    using the normal method
    """
    s = 0
    for i in range(len(p)):
        s += p.coefficients[i] * (x**i)
    return s


def comparison_horner_vs_normal(pol=Polynomial([1, -4, 3, 2]), x=8):
    """
    Función que compara el rendimiento del uso de Horner's method
    con el método normal para evaluación de polinomios.
    """
    t_horners = timeit.timeit(lambda: horners_method(pol, x), number=100000)
    t_normal = timeit.timeit(lambda: normalevaluation(pol, x), number=100000)
    print("Tiempo de ejecución de horners_method():", t_horners)
    print("Tiempo de ejecución de normalevaluation():", t_normal)


def comparison_lagrange_vs_newton_interpolation(
    points_list=[
        (1, 324),
        (3, 2383),
        (5, 6609),
        (6, 324),
        (7, 2383),
        (8, 6609),
        (9, 2383),
        (10, 6609),
    ]
):
    """
    Función que compara la eficiencia del algoritmo de Lagrange vs Newton
    para la interpolación de polinomios.
    """
    t_lagrange = timeit.timeit(
        lambda: get_polynomial_from_points_lagrange(points_list=points_list),
        number=100000,
    )
    t_newton = timeit.timeit(
        lambda: get_polynomial_from_points_newton(points_list=points_list),
        number=100000,
    )
    print("Tiempo de ejecución de interpolación de lagrange:", t_lagrange)
    print("Tiempo de ejecución de interpolación de newton:", t_newton)
