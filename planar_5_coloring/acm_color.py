import igraph as ig
import color
import acm_chromatic_poly as acm

# G = ig.Graph(n=8)
# labels = list(range(8))
# G.add_edges([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
#             (4, 2), (0, 2), (3, 6), (6, 1), (1, 5), (6, 5), (7, 6), (7, 3), (1, 3)])


# Número de vértices en el ciclo exterior
n_outer = 333

# Crear un grafo vacío con n_outer + 1 vértices
g = ig.Graph(n_outer + 1)

# Conectar el vértice central (vértice 0) a cada vértice en el ciclo exterior
for i in range(1, n_outer + 1):
    g.add_edge(0, i)

# Conectar ciclos impares a cada vértice en el ciclo exterior
for i in range(1, n_outer + 1, 3):
    g.add_edge(i, (i % n_outer) + 1)
    g.add_edge((i % n_outer) + 1, ((i + 1) % n_outer) + 1)
    g.add_edge(((i + 1) % n_outer) + 1, i)


acm.plot_graph(g)

acm.greedy_algorithm_coloring(g)

# print(G)

# G2 = color.planar_five_color(G)
# layout = G2.layout("fruchterman_reingold")
# ig.plot(G2, vertex_label=G2.vs['color'], vertex_label_dist=2, layout=layout)
