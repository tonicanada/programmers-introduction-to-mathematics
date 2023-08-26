import sympy as sp


# Householder method to find polynomial roots

x = sp.symbols('x')
f = x**3 + 5 * x**2 + 6 * x

f = sp.sin(x)
f = x**2 + sp.sin(x) -2

p = sp.plot(f, (x, -10, 10), show=True)


def get_pn_householder(f, n):
    fprime = sp.diff(f, x)
    fprime2 = sp.diff(fprime, x)
    print(fprime)
    pn = f * ((fprime2 / (2 * fprime)) - (fprime ** 2 / f))**(n - 1)
    pn = sp.simplify(pn)
    return pn


def get_diff_order_m(f, m):
    while (m > 0):
        f = sp.diff(f, x)
        m = m - 1
    return f


def get_xk_plus_1(f, n):
    xk_plus_1 = x + n * get_diff_order_m(1 / f, n - 1) / \
        get_diff_order_m(1 / f, n)
    return xk_plus_1.simplify()


def get_xk_plus_1_halleys_method(f):
    f_prime = get_diff_order_m(f, 1)
    f_prime2 = get_diff_order_m(f_prime, 1)

    xk_plus_1 = x - (2 * f * f_prime) / (2 * f_prime**2 - f * f_prime2)
    return xk_plus_1.simplify()


def householder_sequence(f, starting_x, householder_order, threshold=1e-12):
    xk_plus_1_fn = get_xk_plus_1(f, householder_order)
    m = float('inf')
    xk_plus_1 = xk_plus_1_fn.subs(x, starting_x).evalf()
    while abs(m) > threshold:
        xk_plus_1 = xk_plus_1_fn.subs(x, xk_plus_1).evalf()
        m = f.subs(x, xk_plus_1).evalf()
        print(m)
    print(xk_plus_1)
    return xk_plus_1


# Check Horner's method for n = 2 is Halleys method

# xk_plus_1 = get_xk_plus_1(f, 2)
# print(xk_plus_1.subs(x, -4))

# xk_plus_1_halleys = get_xk_plus_1_halleys_method(f)
# print(xk_plus_1_halleys.subs(x, -4))


# householder_sequence(f, -2.4, 1)
