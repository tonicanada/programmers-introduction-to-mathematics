import igraph as ig
import color
import time
import timeit
import math
import random
import mug_graphs
import itertools
from datetime import datetime

import sys
sys.path.append('/home/acm/Coding/math/programmers-introduction-to-mathematics')

from secret_sharing.polynomial import Polynomial


# Crea un grafo full_tripartito (que por definición es 3-colorable)
def create_tripartite_graph(n=17):
    # Crea una lista vacía para almacenar las aristas
    edges = []

    # Añade las aristas entre la primera y la segunda partición
    for i in range(n):
        for j in range(n, 2 * n):
            edges.append((i, j))

    # Añade las aristas entre la primera y la tercera partición
    for i in range(n):
        for j in range(2 * n, 3 * n):
            edges.append((i, j))

    # Añade las aristas entre la segunda y la tercera partición
    for i in range(n, 2 * n):
        for j in range(2 * n, 3 * n):
            edges.append((i, j))

    # Crea el grafo
    g = ig.Graph(n=3 * n, edges=edges)

    return g


def create_random_tripartite_graph(n):
    edges = []

    # Añade las aristas entre la primera y la segunda partición de forma aleatoria
    for i in range(n):
        for j in range(n, 2 * n):
            if random.random() < 0.5:
                edges.append((i, j))

    # Añade las aristas entre la primera y la tercera partición de forma aleatoria
    for i in range(n):
        for j in range(2 * n, 3 * n):
            if random.random() < 0.5:
                edges.append((i, j))

    # Añade las aristas entre la segunda y la tercera partición de forma aleatoria
    for i in range(n, 2 * n):
        for j in range(2 * n, 3 * n):
            if random.random() < 0.5:
                edges.append((i, j))

    # Crea el grafo
    g = ig.Graph(n=3 * n, edges=edges)

    return g


def get_graph_example(tipo):
    # 3-colorable examples
    if tipo == 'Petersen':
        g = ig.Graph.Famous('Petersen')
    elif tipo == 'Custom_1_connected':
        g = ig.Graph(n=8)
        g.add_edges([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
                     (4, 2), (0, 2), (3, 6), (6, 1), (1, 5), (6, 5), (7, 6), (7, 3), (1, 3)])
    # 2-colorable examples
    elif tipo == 'Tree':
        g = ig.Graph.Tree(n=10, children=3)
    elif tipo == 'FullBipartite':
        g = ig.Graph.Full_Bipartite(3, 4)
    elif tipo == 'Ring':
        g = ig.Graph.Ring(n=4, circular=True)
    elif tipo == 'Rejilla2D':
        g = ig.Graph.Lattice(dim=[4, 4], circular=False)
    elif tipo == 'Disconnected_2_clusters':
        g = ig.Graph(edges=[(0, 1), (1, 2), (2, 3), (3, 0), (4, 5),
                     (5, 6), (6, 7), (7, 8), (8, 9), (9, 4)])
    # 3-colorable examples
    elif tipo == 'FullTripartite':
        g = create_tripartite_graph(5)
    elif tipo == 'RandomTripartiteGraph':
        g = create_random_tripartite_graph(10)
    # Full connected graphs
    elif tipo == 'K5':
        g = ig.Graph.Full(6)
    return g


def plot_graph(graph, vertex_label=None):
    if vertex_label == None:
        v_label_index = [v.index for v in graph.vs]
        ig.plot(graph, vertex_label=v_label_index,
                vertex_label_dist=2, bbox=(500, 500), vertex_size=5)
    else:
        ig.plot(graph, vertex_label=vertex_label,
                vertex_label_dist=2, bbox=(500, 500), vertex_size=5)


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
    print("Tiempo de ejecución sin memoización:", t_withoutmemo)
    print("Tiempo de ejecución con memoización:", t_memo)


def get_cycles(graph):
    cycles = []
    for start_vertex in range(graph.vcount()):
        paths = graph.get_all_simple_paths(start_vertex)
        for path in paths:
            if len(path) > 2 and graph.are_connected(path[0], path[-1]):
                cycles.append(path)
    return cycles


def check_graph_2_colorable(graph):
    return graph.is_bipartite()


def color_2_colorable_graph(graph, colors=[0, 1]):
    # Check if graph is 2-colorable
    if not check_graph_2_colorable(graph):
        print("Graph is not 2-colorable")
        return False
    # Check if graph is not connected, and check for clusters
    clusters = graph.clusters()
    for cluster in clusters:
        start = cluster[0]
        graph.vs[start]['color'] = 0
        dfs = graph.dfs(start)[0]
        for i in dfs:
            if i == start:
                continue
            v = graph.vs[i]
            neighbor_colors = set(w['color'] for w in v.neighbors())
            v['color'] = [j for j in colors if j not in neighbor_colors][0]
    return graph


def greedy_algorithm_coloring(graph, starting_color=0):
    degrees = graph.degree()
    max_degree = max(degrees)
    # We know an upper bound for any graph, X(G) <= max(degree) + 1
    colors = [starting_color + i for i in range(max_degree + 1)]
    print("greedy", colors)
    used_colors = []
    start = 0
    graph.vs[0]['color'] = colors[0]
    used_colors.append(colors[0])
    for v_index in range(graph.vcount()):
        if v_index == start:
            continue
        v = graph.vs[v_index]
        neighbor_colors = set(w['color'] for w in v.neighbors())
        v['color'] = [j for j in colors if j not in neighbor_colors][0]
    plot_graph(graph, vertex_label=graph.vs['color'])
    print(list(set(graph.vs['color'])))
    used_colors = len(list(set(graph.vs['color'])))
    return [graph, used_colors]


def dfs_not_colored(graph, start_vertex, colors):
    neighbors = start_vertex.neighbors()
    neighbor_colors = set(w['color'] for w in neighbors)
    start_vertex['color'] = [j for j in colors if j not in neighbor_colors][0]
    for nw in neighbors:
        if nw['color'] == None:
            dfs_not_colored(graph, nw, colors)
            break


def color_3_colorable_graph_at_most_4_sqrt_n(graph):
    used_colors = 0
    n = graph.vcount()
    sqrt_n = math.ceil(math.sqrt(n))
    colors = list(range(4 * sqrt_n))
    graph.vs['color'] = [None] * n
    used_colors = 0

    if graph.is_bipartite():
        print("bipartite")
        graph = color_2_colorable_graph(graph)

    clusters = graph.clusters()

    for cluster in clusters:
        subgraph = graph.subgraph(cluster)

        while len(subgraph.vs.select(lambda v: v['color'] == None)) > 0:
            vertex_degree_at_least_sqrt_n = subgraph.vs.select(
                lambda v: subgraph.degree(v) >= sqrt_n and v['color'] == None)
            if len(vertex_degree_at_least_sqrt_n) > 0:
                color_set = colors[used_colors:used_colors + 3]
                print(color_set)
                used_colors = used_colors + 3
                current_vertex = vertex_degree_at_least_sqrt_n[0]
                current_vertex['color'] = color_set[0]
                neighbors = current_vertex.neighbors()
                for nw in neighbors:
                    nw_nw_colors = set(w['color'] for w in nw.neighbors())
                    nw['color'] = [j for j in color_set[1:] if j not in nw_nw_colors][0]
            else:
                vertex_degree_at_most_sqrt_n = subgraph.vs.select(lambda v: v['color'] == None)
                current_vertex = vertex_degree_at_most_sqrt_n[0]
                dfs_not_colored(subgraph, current_vertex, colors[used_colors:])

        for i, vertex_index in enumerate(cluster):
            graph.vs[vertex_index]["color"] = subgraph.vs[i]["color"]

    print(colors)
    print(list(set(graph.vs['color']))
          )
    plot_graph(graph, vertex_label=graph.vs['color'])
    return graph


def generate_and_save_3_colorable_graph(tipo_mug="MUG11a", k=25):
    mug = mug_graphs.get_mug_graph(tipo_mug)
    g = mug_graphs.graph_generator_3_colored(mug, k)

    g.write_graphml(
        f"./sample_graphs/{datetime.today().strftime('%Y%m%d_%H%M%S')}_{tipo_mug}_{k}.graphml")


def get_sample_strongly_connected(tipo='strongly_connected'):
    # Primer grafo dirigido, NO fuertemente conectado
    g1 = ig.Graph(directed=True)
    g1.add_vertices(31)  # Añadimos 31 vértices
    g1.add_edges([(i, i + 1) for i in range(30)])  # Añadimos 30 aristas que van desde i hasta i + 1

    # Segundo grafo dirigido, NO fuertemente conectado
    g2 = ig.Graph(directed=True)
    g2.add_vertices(31)  # Añadimos 31 vértices
    # Añadimos aristas de manera que no todos los vértices estén conectados
    g2.add_edges([(i, (i * 2) % 31) for i in range(31)])

    # Tercer grafo dirigido, FUERTEMENTE conectado
    g3 = ig.Graph(directed=True)
    g3.add_vertices(31)  # Añadimos 31 vértices
    # Añadimos 31 aristas que forman un ciclo
    g3.add_edges([(i, i + 1) for i in range(30)] + [(30, 0)])

    # Cuarto grafo dirigido, FUERTEMENTE conectado
    g4 = ig.Graph(directed=True)
    g4.add_vertices(31)  # Añadimos 31 vértices
    g4.add_edges([(i, j) for i in range(31) for j in range(i + 1, 31)] + [(j, i) for i in range(31)
                 for j in range(i + 1, 31)])  # Cada vértice está conectado con todos los demás en ambas direcciones

    if tipo == 'strongly_connected':
        return [g3, g4]
    elif tipo == 'not_strongly_connected':
        return [g1, g2]


def check_strongly_connected(graph):
    vertex_list = list(range(graph.vcount()))
    combinations = list(itertools.combinations(vertex_list, 2))
    for comb in combinations:
        path1 = graph.get_shortest_paths(comb[0], to=comb[1])
        path2 = graph.get_shortest_paths(comb[1], to=comb[0])
        if (len(path1[0]) == 0) or (len(path2[0]) == 0):
            return False
    return True


def get_sample_dag_graph():
    g1 = ig.Graph(directed=True)

    # Agregar 31 vértices al grafo
    g1.add_vertices(31)

    # Agregar aristas al grafo. Recuerda que los índices de los vértices en igraph empiezan en 0
    edges = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8), (7, 9), (8, 10),
             (9, 11), (10, 12), (11, 13), (12, 14), (13, 15), (14, 16), (15, 17), (16, 18),
             (17, 19), (18, 20), (19, 21), (20, 22), (21, 23), (22, 24), (23, 25), (24, 26),
             (25, 27), (26, 28), (27, 29), (28, 30)]

    g1.add_edges(edges)

    g2 = ig.Graph(directed=True)

    # Agregar 31 vértices al grafo
    g2.add_vertices(31)

    # Agregar aristas al grafo. Recuerda que los índices de los vértices en igraph empiezan en 0
    edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6), (3, 7), (3, 8), (4, 9), (4, 10),
             (5, 11), (5, 12), (6, 13), (6, 14), (7, 15), (8, 16), (9, 17), (10, 18),
             (11, 19), (12, 20), (13, 21), (14, 22), (15, 23), (16, 24), (17, 25), (18, 26),
             (19, 27), (20, 28), (21, 29), (22, 30)]

    g2.add_edges(edges)

    g3 = ig.Graph(directed=True)

    # Agregar 31 vértices al grafo
    g3.add_vertices(31)

    # Creamos una lista de todas las posibles aristas, respetando la propiedad acíclica
    possible_edges = [(i, j) for i in range(30) for j in range(i+1, 31)]

    # Fijamos random seed
    random.seed(26)
    
    # Barajamos la lista de posibles aristas
    random.shuffle(possible_edges)

    # Seleccionamos las primeras 45 aristas
    edges = possible_edges[:45]

    # Agregar aristas al grafo
    g3.add_edges(edges)

    return [g1, g2, g3]


def get_topological_order(graph):
    in_degrees = graph.degree(type="in")
    n = graph.vcount()
    order = []
    queue = [i for i in range(n) if in_degrees[i] == 0]

    while queue:
        current_index = queue.pop(0)
        order.append(current_index)
        neighbors = graph.neighbors(current_index, mode="out")
        for neighbor in neighbors:
            in_degrees[neighbor] -= 1
            if in_degrees[neighbor] == 0:
                queue.append(neighbor)

    if len(order) != n:
        print("El grafo contiene un ciclo!")
    else:
        print(order)




g1 = ig.Graph.Read_GraphML("./planar_5_coloring/sample_graphs/20230802_225201_MUG12c_2.graphml")
g2 = ig.Graph.Read_GraphML("./planar_5_coloring/sample_graphs/20230802_225349_MUG9_2.graphml")
g3 = ig.Graph.Read_GraphML("./planar_5_coloring/sample_graphs/20230802_225655_MUG11a_3.graphml")

tree = get_graph_example('Tree')
# g = graph_initial = g2.disjoint_union(g3)
# g = color_3_colorable_graph_at_most_4_sqrt_n(tree)


# g = get_sample_strongly_connected(tipo='strongly_connected')[0]
# plot_graph(g)
# a = check_strongly_connected(g)
# print(a)


g = get_sample_dag_graph()[0]
plot_graph(g)
get_topological_order(g)

