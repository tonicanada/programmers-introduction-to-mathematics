import igraph as ig
# import acm_chromatic_poly as acm
import color
import random

def get_mug_graph(tipo):
    if tipo == "MUG10":
        g = ig.Graph(n=10)
        g.add_edges([(0, 1), (0, 2), (1, 2), (0, 3), (1, 4), (2, 5), (3, 4), (3, 5), (4, 6),
                     (6, 5), (2, 7), (1, 8), (6, 9), (5, 8), (4, 7), (7, 9), (8, 9), (0, 9)])
    if tipo == "MUG9":
        g = ig.Graph(n=9)
        g.add_edges([(0, 1), (0, 2), (1, 2), (1, 3), (3, 5),
                     (2, 4), (4, 8), (5, 6), (6, 7), (7, 8), (1, 7), (2, 6), (3, 4), (0, 5), (5, 8), (0, 8)])
    if tipo == "MUG11a":
        g = ig.Graph(n=11)
        g.add_edges([(0, 1), (1, 2), (2, 10), (10, 9), (9, 8), (8, 5), (5, 0), (0, 4), (4, 7),
                    (7, 8), (1, 3), (3, 5), (5, 6), (3, 6), (3, 7), (6, 4), (4, 2), (7, 10), (1, 9), (6, 9)])
    if tipo == "MUG11b":
        g = ig.Graph(n=11)
        g.add_edges([(0, 1), (1, 2), (2, 10), (10, 9), (9, 0), (0, 5), (2, 3), (1, 4), (3, 4),
                    (4, 5), (0, 6), (2, 8), (7, 8), (6, 7), (8, 10), (9, 6), (3, 6), (4, 7), (5, 8)])
    if tipo == "MUG12a":
        g = ig.Graph(n=12)
        g.add_edges([(0, 1), (1, 2), (2, 8), (8, 11), (11, 10), (10, 6), (6, 0), (0, 3), (3, 4),
                    (4, 5), (5, 2), (3, 10), (6, 9), (9, 8), (3, 7), (7, 9), (9, 5), (1, 7), (7, 10), (1, 11), (4, 11), (4, 9)])
    if tipo == "MUG12b":
        g = ig.Graph(n=12)
        g.add_edges([(0, 1), (0, 3), (3, 10), (10, 11), (11, 7), (7, 1), (1, 2), (2, 4), (4, 3), (0, 8),
                    (0, 5), (0, 7), (1, 9), (2, 5), (5, 8), (5, 10), (5, 6), (8, 9), (9, 6), (6, 7), (8, 11), (4, 8)])
    if tipo == "MUG12c":
        g = ig.Graph(n=12)
        g.add_edges([(0, 2), (2, 1), (0, 3), (3, 7), (7, 9), (9, 10), (10, 11), (11, 8), (8, 4), (4, 1),
                    (5, 3), (3, 2), (5, 7), (7, 10), (2, 4), (4, 6), (10, 8), (8, 6), (0, 11), (1, 9), (5, 6)])
    return g


def get_random_edge_with_some_vertex_deg_le_3(graph):
    vertex_deg_le_3 = graph.vs.select(_degree_le=3)
    n = len(vertex_deg_le_3)
    choice = random.choice(list(range(n)))
    vertex_deg_le_3 = vertex_deg_le_3[choice]

    edge_indices = graph.incident(vertex_deg_le_3)
    edge = graph.es[edge_indices][0]

    if edge.source == vertex_deg_le_3.index:
        i = edge.source
        j = edge.target
    else:
        i = edge.target
        j = edge.source
    return (i, j)


def flatten(lst):
    result = []
    for i in lst:
        if isinstance(i, list):
            result.extend(flatten(i))
        else:
            result.append(i)
    return result


def flat_list(original_list):
    new_list = []
    for element in original_list:
        if element is None:
            new_list.append([None])
        elif type(element) == int:
            new_list.append([element])
        elif len(element) == 1:
            new_list.append(flatten(element))
        elif len(element) >= 2:
            new_list.append(flatten(element))
    return new_list


def graph_generator_3_colored(graph_initial, k):

    n = graph_initial.vcount()

    for iter in range(k):

        n = graph_initial.vcount()
        graph_initial.vs['old_index'] = list(range(n))

        if 'old_index_mug' in graph_initial.vertex_attributes():
            del graph_initial.vs['old_index_mug']

        (i, j) = get_random_edge_with_some_vertex_deg_le_3(graph_initial)

        tipos_mug = ["MUG9", "MUG10", "MUG11a", "MUG11b", "MUG12a", "MUG12b", "MUG12c"]
        tipo = random.choice(tipos_mug)
        print(tipo)
        mug = get_mug_graph(tipo)
        m = mug.vcount()
        mug.vs['old_index_mug'] = list(range(m))

        (x, y) = get_random_edge_with_some_vertex_deg_le_3(mug)

        graph_initial = graph_initial.disjoint_union(mug)

        vi = graph_initial.vs.select(lambda v: v['old_index'] == i)[0]
        vj = graph_initial.vs.select(lambda v: v['old_index'] == j)[0]

        graph_initial.delete_edges((vi.index, vj.index))
        mug.delete_edges((x, y))

        vy = graph_initial.vs.select(lambda v: v['old_index_mug'] == y)[0]
        graph_initial.add_edges([(vj.index, vy.index)])
        vx = graph_initial.vs.select(lambda v: v['old_index_mug'] == x)[0]
        color.merge_two(graph_initial, vx, vi)

    print("clusters", len(graph_initial.clusters()))

    return graph_initial
