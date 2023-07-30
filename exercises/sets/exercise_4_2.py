# 4.12. Continuing the previous exercise, the non-existence of Steiner systems for some
# choices of n suggests a modified problem of finding a minimal size family F of size-k
# subsets such that every t-size subset is in at least one set in F . For (n, k, t) arbitrary, find
# a lower bound on the size of F . Try to come up with an algorithm that gets close to this
# lower bound for small values of k, t.

from itertools import combinations, permutations


def get_n_choose_k_combinations(n, k):
    combinaciones = list(combinations(range(1, n + 1), k))
    return combinaciones


def check_if_subarray_elements_included_in_array(subarray, array):
    if all(elem in array for elem in subarray):
        return True
    else:
        return False


def contiene_elementos(lista1, lista2):
    for elemento in lista2:
        if elemento in lista1:
            return True
    return False


def get_lower_bound_steiner(n, k, t):
    comb_n_k = get_n_choose_k_combinations(n, k)
    comb_n_t = get_n_choose_k_combinations(n, t)
    dict_n_t = {}
    for comb in comb_n_t:
        dict_n_t[comb] = False
    dict_n_k = {}
    for comb in comb_n_k:
        dict_n_k[comb] = []

    result = []
    idx_to_drop = []

    for comb in comb_n_k:
        c = list(combinations(comb, t))
        c_bool = list(map(lambda x: True if dict_n_t[x] else False, c))
        if all(not x for x in c_bool):
            for c1 in c:
                dict_n_t[c1] = True
                if comb not in result:
                    result.append(comb)

    for key in dict_n_t:
        if not dict_n_t[key]:
            print("Steiner System doesn't found for given conditions")
            return
    print(result)
    return result


get_lower_bound_steiner(7, 3, 2)
