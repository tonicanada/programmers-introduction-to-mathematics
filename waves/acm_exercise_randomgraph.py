import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import math


# 12.9. Implement the algorithm presented in the chapter to generate a random graph on
# n vertices with edge
# √ probability 1/2, and a planted clique of size k. For the rest of this
# exercise fix k = ⌈ n log n⌉. Determine the average degree of a vertex that is in the plant,
# and the average degree of a vertex that is not in the plant, and use that to determine a rule
# for deciding √
# if a vertex is in the clique. Implement this rule for finding planted cliques of
# size at least n log n with high probability, where n = 1000.


# Configurando la semilla
random.seed(42)


def generate_random_graph(n):
    adj = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if i == j:
                adj[i, j] = 0
            else:
                adj[i, j] = random.randint(0, 1)
                adj[j, i] = adj[i, j]
    return adj


def generate_k_clique(adj, k):
    n = len(adj)
    if (k > n):
        print(f"k {k} no puede ser mayor a n {n}")
    vertex_list = list(range(n))
    clique_k_vertex = random.sample(vertex_list, k)
    for i in range(k):
        for j in range(i, k):
            u = clique_k_vertex[i]
            v = clique_k_vertex[j]
            if u == v:
                adj[u, v] = 0
            else:
                adj[u, v] = 1
                adj[v, u] = 1

    # Calculamos el promedio de conectividad de un vertice que está dentro del clique
    mean_vertex_clique = get_mean_connectivity(adj, clique_k_vertex)
    print(f"El promedio de conectividad de un vértice del clique es de {mean_vertex_clique}")

    # Calculamos el promedio de conectividad de un vertice que no está dentro del clique
    set_clique = set(clique_k_vertex)
    set_general = set(vertex_list)
    set_not_clique = set_general - set_clique
    not_clique_vertex = list(set_not_clique)
    mean_non_vertex_clique = get_mean_connectivity(adj, not_clique_vertex)
    print(
        f"El promedio de conectividad de un vértice que no es del clique es de {mean_non_vertex_clique}")

    return adj, set(clique_k_vertex)


def get_mean_connectivity(adj, vertex_list):
    rows = adj[vertex_list, :]
    rows_connectivity = np.mean(rows, axis=1) / 2
    rows_connectivity = np.mean(rows_connectivity)
    return rows_connectivity


def plot_graph(adj):
    g = nx.from_numpy_array(adj)
    pos = nx.spring_layout(g, seed=42)
    nx.draw(g, pos=pos, with_labels=True, node_color='skyblue',
            node_size=700, font_weight='bold', edge_color='gray')
    plt.show()


def detect_cliques_with_eigenvector(adj):
    n = len(adj)
    sqrt_n = int(math.sqrt(n))
    print("sqrt", sqrt_n)
    _, eigenvectors = np.linalg.eig(adj)
    eig_2 = eigenvectors[:, 1]
    t = np.argsort(np.abs(eig_2))[::-1]
    t = set(t[:sqrt_n])
    vertex_clique = []
    for i in range(n):
        row = adj[i, :]
        index_connected_to_i = set(np.where(row == 1)[0].tolist())
        t_intersection = t.intersection(index_connected_to_i)
        percentage = (len(t_intersection) / len(t))
        if percentage >= 3 / 4:
            vertex_clique.append(i)
    return set(vertex_clique)


def exercise_12_9():
    n = 1000
    k = int(math.sqrt(n * math.log(n)))
    adj = generate_random_graph(n)
    adj, clique_k_vertex = generate_k_clique(adj, k)
    print(adj)
    print(clique_k_vertex)


def exercise_12_10():
    n = 1000
    k = 10 * int(math.sqrt(n))
    adj = generate_random_graph(n)
    adj, clique_planted_k_vertex = generate_k_clique(adj, k)
    detected_clique = detect_cliques_with_eigenvector(adj)
    check_clique = clique_planted_k_vertex.intersection(detected_clique)
    print(
        f"El clique detectado incluye un {len(check_clique)/len(clique_planted_k_vertex)*100}% del clique plantado")


# exercise_12_9()
exercise_12_10()
