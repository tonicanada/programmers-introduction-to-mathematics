import numpy as np
import random


# 12.13. Implement the Gram-Schmidt algorithm using the following method for finding
# vectors not in the span of a partial basis: choose a vector with random entries between
# zero and one, repeating until you find one that works. How often does it happen that you
# have to repeat? Can you give an explanation for this?


random.seed(42)


def exercise_12_3(n):
    random_vector = np.random.rand(n)
    random.seed(None)
    prod = 1
    iterations = 0
    margin = 1e-2
    while prod > margin:
        new_random_vector = np.random.rand(n)
        prod = np.dot(random_vector, new_random_vector)
        iterations += 1
        print(prod)
    print(iterations)


exercise_12_3(3)
