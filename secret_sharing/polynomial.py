from itertools import zip_longest
import numpy as np


def strip(L, elt):
    """Strip all copies of elt from the end of the list.

    Arguments:
        L: a list (an indexable, sliceable object)
        elt: the object to be removed

    Returns:
        a slice of L with all copies of elt removed from the end.
    """
    if len(L) == 0:
        return L

    i = len(L) - 1
    while i >= 0 and L[i] == elt:
        i -= 1

    return L[: i + 1]


class Polynomial(object):
    """A class representing a polynomial as a list of coefficients with no
    trailing zeros.

    A degree zero polynomial corresponds to the empty list of coefficients,
    and is provided by this module as the variable ZERO.

    Polynomials override the basic arithmetic operations.
    """

    def __init__(self, coefficients):
        """Create a new polynomial.

        The caller must provide a list of all coefficients of the
        polynomial, even those that are zero. E.g.,
        Polynomial([0, 1, 0, 2]) corresponds to f(x) = x + 2x^3.
        """
        self.coefficients = strip(coefficients, 0)
        self.indeterminate = "x"

    def add(self, other):
        new_coefficients = [sum(x) for x in zip_longest(self, other, fillvalue=0.0)]
        return Polynomial(new_coefficients)

    def __add__(self, other):
        return self.add(other)

    def multiply(self, other):
        new_coefficients = [0] * (len(self) + len(other) - 1)

        for i, a in enumerate(self):
            for j, b in enumerate(other):
                new_coefficients[i + j] += a * b

        return Polynomial(strip(new_coefficients, 0))

    def degree(self):
        return len(self.coefficients) - 1

    def long_division(self, divisor):
        dividend = Polynomial(self.coefficients)

        result_degree = len(self.coefficients) - len(divisor.coefficients)
        result = {
            "quotient": np.zeros(result_degree + 1),
            "remainder": 0
        }

        counter = result_degree + 1
        for idx in range(result_degree + 1):
            # Find quotient
            a = dividend.coefficients[-1] / divisor.coefficients[-1]
            result['quotient'][idx] = a
            mult_list = np.flip(result['quotient'])[:counter]
            mult_poly = Polynomial(mult_list)
            # Multiply
            mult = mult_poly * divisor
            # Substract
            dividend = dividend.add(-mult)
            if len(dividend.coefficients) == 0:
                break
            counter = counter - 1
        result['quotient'] = np.flip(result['quotient'])
        result['quotient'] = Polynomial(result['quotient'])
        result['remainder'] = dividend
        return result

    # def extended_euclidean_algorithm(self, other);

    def power(self, exponent):
        base = Polynomial([1])
        # if exponent == 0:
        #     return Polynomial([1])
        for i in range(exponent):
            base *= self
        return base

    def __mul__(self, other):
        return self.multiply(other)

    def __len__(self):
        """len satisfies len(p) == 1 + degree(p)."""
        return len(self.coefficients)

    def __repr__(self):
        return " + ".join(
            [
                "%s %s^%d" % (a, self.indeterminate, i) if i > 0 else "%s" % a
                for i, a in enumerate(self.coefficients)
            ]
        )

    def evaluate_at(self, x):
        """Evaluate a polynomial at an input point.

        Uses Horner's method, first discovered by Persian mathematician
        Sharaf al-Dīn al-Ṭūsī, which evaluates a polynomial by minimizing
        the number of multiplications.
        """
        theSum = 0

        for c in reversed(self.coefficients):
            theSum = theSum * x + c

        return theSum

    def __iter__(self):
        return iter(self.coefficients)

    def __neg__(self):
        return Polynomial([-a for a in self])

    def __sub__(self, other):
        return self + (-other)

    def __call__(self, *args):
        return self.evaluate_at(args[0])

    def __pow__(self, value):
        return self.power(value)


ZERO = Polynomial([])


def euclidean_algorithm(poly1, poly2, q_list):
    # Base case
    if (poly2.coefficients == []):
        res = Polynomial(q_list[-2])
        return res
    result = poly1.long_division(poly2)
    q_list.append(result['remainder'].coefficients)
    return euclidean_algorithm(poly2, result['remainder'], q_list)


# Test long division
def test_long_division(dividend, divisor):
    result = Polynomial(dividend).long_division(Polynomial(divisor))
    quotient = result['quotient']
    remainder = result['remainder']
    print("QUOTIENT", quotient)
    print("REMAINDER", remainder)
    print((quotient * divisor + remainder).coefficients == dividend)


# Test euclidean algorithm
def test_eucliedan_algorithm():
    dividend = Polynomial([-3, 1]) * Polynomial([-5, 1]) * Polynomial([-10, 1])
    print("DIVIDEND", dividend)
    divisor = Polynomial([-3, 1])
    print("DIVISOR", divisor)
    print("DIV", dividend.long_division(divisor))
    result = euclidean_algorithm(dividend, divisor)
    print(str(result))


# test_long_division([-150, 95, -18, 1], [-3, 1])
# test_long_division([150, 10], [-3, 1])
# test_long_division([5, -11, -7, 4], [5, 4]),
# test_long_division([2, 0, 6, 0, 1], [5, 0, 1])
# test_long_division([-1, 2, -5, 3], [1, -3, 2])
# test_long_division([0, -0.25], [4])

# test_eucliedan_algorithm()
# test_eucliedan_algorithm()

poly1 = [-1, 3, -3, 1]
poly2 = [4, -5, 2]
res = euclidean_algorithm(Polynomial(poly1), Polynomial(poly2), [])
a = Polynomial(poly1).long_division(res)
b = Polynomial(poly2).long_division(res)
print(res)
print(b)
print(a)




# poly1 = Polynomial([4, -5, 1])
# poly2 = Polynomial([-9, 9])
# print(poly1.long_division(poly2))
