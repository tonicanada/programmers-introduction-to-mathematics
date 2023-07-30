import igraph as ig
import color
import time
import timeit

import sys
sys.path.append('/home/acm/Coding/math/programmers-introduction-to-mathematics')

from secret_sharing.polynomial import Polynomial


# Sample graph
# G = ig.Graph(n=8)
# labels = list(range(8))
# G.add_edges([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
#             (4, 2), (0, 2), (3, 6), (6, 1), (1, 5), (6, 5), (7, 6), (7, 3), (1, 3)])

# G.vs['old_index'] = list(range(8))


# # Linear graph
# G = ig.Graph(n=5)
# labels = list(range(5))
# G.add_edges([(0, 1), (1, 2), (2, 3), (3, 4)])

# G.vs['old_index'] = list(range(5))


# Petersen
G = ig.Graph.Famous('Petersen')


# layout = G.layout("fruchterman_reingold")

# ig.plot(G, vertex_label_dist=2, bbox=(500, 500))


def remove_edges(graph):
    if len(graph.get_edgelist()) == 0:
        return graph

    ig.plot(graph, vertex_label=labels, vertex_label_dist=2, bbox=(500, 500), layout=layout)
    print(graph.get_edgelist())
    edge = graph.get_edgelist()[0]
    graph.delete_edges(edge)
    print(graph.get_edgelist())
    remove_edges(graph)


def contract_edges(graph):
    if len(graph.get_edgelist()) == 0:
        return graph
    ig.plot(graph, vertex_label=labels, vertex_label_dist=2, bbox=(500, 500), layout=layout)
    print(graph.get_edgelist())
    edge_id = graph.get_edgelist()[0]

    v1 = graph.get_edgelist()[0][0]
    v2 = graph.get_edgelist()[0][1]

    v1 = graph.vs[v1]
    v2 = graph.vs[v2]

    color.merge_two(graph, v1, v2)
    graph.simplify()
    print(graph.get_edgelist())
    contract_edges(graph)


def graph_id(graph):
    # Obtener la matriz de adyacencia
    adjacency_matrix = graph.get_adjacency()

    # Convertir la matriz a una cadena
    string_representation = ''.join(str(e) for row in adjacency_matrix for e in row)

    return string_representation


def chromatic_polynomial_withoutmemo(graph):

    # Base case: if the graph has no edges, return k^n
    if graph.ecount() == 0:
        poly1_arr = [0] * (graph.vcount() + 1)
        poly1_arr[-1] = 1
        poly = Polynomial(poly1_arr)
        return poly

    # Choose an arbitrary edge
    edge = graph.es[0]
    vertices = edge.tuple

    # Recursion step: compute the polynomials for G-e and G/e
    # G - e
    graph_minus_e = graph.copy()
    graph_minus_e.delete_edges(edge)
    P_graph_minus_e = chromatic_polynomial_withoutmemo(graph_minus_e)

    # G / e
    graph_over_e = graph.copy()
    v1 = graph_over_e.vs[vertices[0]]
    v2 = graph_over_e.vs[vertices[1]]
    color.merge_two(graph_over_e, v1, v2)
    graph_over_e.simplify(multiple=True, loops=True, combine_edges=None)
    P_graph_over_e = chromatic_polynomial_withoutmemo(graph_over_e)

    # The chromatic polynomial of G is P(G-e, k) - P(G/e, k)
    return P_graph_minus_e - P_graph_over_e


def chromatic_polynomial(graph, memo):

    # Check memo
    if graph_id(graph) in memo:
        return memo[graph_id(graph)]

    # Base case: if the graph has no edges, return k^n
    if graph.ecount() == 0:
        poly1_arr = [0] * (graph.vcount() + 1)
        poly1_arr[-1] = 1
        poly = Polynomial(poly1_arr)
        return poly

    # Choose an arbitrary edge
    edge = graph.es[0]
    vertices = edge.tuple

    # Recursion step: compute the polynomials for G-e and G/e
    # G - e
    graph_minus_e = graph.copy()
    graph_minus_e.delete_edges(edge)
    P_graph_minus_e = chromatic_polynomial(graph_minus_e, memo)
    memo[graph_id(graph_minus_e)] = P_graph_minus_e

    # G / e
    graph_over_e = graph.copy()
    v1 = graph_over_e.vs[vertices[0]]
    v2 = graph_over_e.vs[vertices[1]]
    color.merge_two(graph_over_e, v1, v2)
    graph_over_e.simplify(multiple=True, loops=True, combine_edges=None)
    P_graph_over_e = chromatic_polynomial(graph_over_e, memo)
    memo[graph_id(graph_over_e)] = P_graph_over_e

    # The chromatic polynomial of G is P(G-e, k) - P(G/e, k)
    return P_graph_minus_e - P_graph_over_e


def chromatic_polynomial_heuristic(graph, memo):
    """
    Esta versión de la función utiliza la heurística de escoger el edge con mayor degree.
    La eliminación o contracción de un borde así podría tener un impacto más significativo en la reducción del tamaño del grafo.
    """
    # Check memo
    if graph_id(graph) in memo:
        return memo[graph_id(graph)]

    # Base case: if the graph has no edges, return k^n
    if graph.ecount() == 0:
        poly1_arr = [0] * (graph.vcount() + 1)
        poly1_arr[-1] = 1
        poly = Polynomial(poly1_arr)
        return poly

    # Choose an edge with the highest degree
    max_degree = -1
    for edge in graph.es:
        degree = graph.degree(edge.source) + graph.degree(edge.target)
        if degree > max_degree:
            max_degree = degree
            chosen_edge = edge

    vertices = chosen_edge.tuple

    # Recursion step: compute the polynomials for G-e and G/e
    # G - e
    graph_minus_e = graph.copy()
    graph_minus_e.delete_edges(chosen_edge)
    P_graph_minus_e = chromatic_polynomial_heuristic(graph_minus_e, memo)
    memo[graph_id(graph_minus_e)] = P_graph_minus_e

    # G / e
    graph_over_e = graph.copy()
    v1 = graph_over_e.vs[vertices[0]]
    v2 = graph_over_e.vs[vertices[1]]
    color.merge_two(graph_over_e, v1, v2)
    graph_over_e.simplify(multiple=True, loops=True, combine_edges=None)
    P_graph_over_e = chromatic_polynomial_heuristic(graph_over_e, memo)
    memo[graph_id(graph_over_e)] = P_graph_over_e

    # The chromatic polynomial of G is P(G-e, k) - P(G/e, k)
    return P_graph_minus_e - P_graph_over_e


def comparison_functions():
    g = ig.Graph.Famous('Petersen')

    t_withoutmemo = timeit.timeit(
        lambda: chromatic_polynomial_withoutmemo(g),
        number=20,
    )
    t_memo = timeit.timeit(
        lambda: chromatic_polynomial(g, {}),
        number=20,
    )
    t_memo_heuristic = timeit.timeit(
        lambda: chromatic_polynomial_heuristic(g, {}),
        number=20,
    )
    print("Tiempo de ejecución sin memoización:", t_withoutmemo)
    print("Tiempo de ejecución con memoización:", t_memo)
    print("Tiempo de ejecución con memoización y heurística:", t_memo_heuristic)


m = {}
a = chromatic_polynomial(G, m)
print(a)


comparison_functions()
