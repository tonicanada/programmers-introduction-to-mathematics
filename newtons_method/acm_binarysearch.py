import math

import sys
sys.path.append('/home/acm/Coding/math/programmers-introduction-to-mathematics')

from secret_sharing.polynomial import Polynomial


poly1 = Polynomial([-2, 1]) * Polynomial([3, 1]) * Polynomial([5, 1]) * Polynomial([-100, 1])
poly2 = Polynomial([-1, 7, -4, 1])

a = 2.1
b = 1.9
c = (b - a) / 2 + a
# print(c)
# print(poly1.evaluate_at(a))
# print(poly1.evaluate_at(b))
# print(poly1.evaluate_at(c))


def binarysearch_roots(poly, a, b):
    m = float('inf')
    threshold = 1e-12
    counter = 0
    interval = ((b - a) / 2) / 100
    while abs(m) > threshold:
        fa = poly.evaluate_at(a)
        fb = poly.evaluate_at(b)
        c = ((b - a) / 2) + a

        m = poly.evaluate_at(c)
        if m * fa < 0:
            b = c
        elif m * fb < 0:
            a = c
        elif m * fb > 0 and m * fa > 0:
            min_val = min(abs(fa), abs(fb))
            if fa > 0:
                if min_val == fa:
                    temp = b
                    a = a - abs(interval)
                    b = temp
                else:
                    temp = b
                    b = b + abs(interval)
                    a = temp
            else:
                if min_val == fa:
                    temp = a
                    a = a + abs(interval)
                    b = temp
                else:
                    temp = b
                    b = b - abs(interval)
                    a = temp
        elif fa == 0:
            print("root", a)
            return a
        elif fb == 0:
            print("root", b)
            return b

        print("XXXXX", m, a, b)
    print(m, c)


binarysearch_roots(poly1, -10, -4)
