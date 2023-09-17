import networkx as nx
import matplotlib.pyplot as plt

# Ejercicio 12.8
# Escribe un programa que detecte ciclos impares en grafos,
# de esta manera podemos saber si es bipartito o no

n = 7  # Puedes cambiar este valor
# 0.5 es la probabilidad de que exista una arista entre dos nodos
g = nx.gnp_random_graph(n, 0.5, seed=42)

# g = nx.bipartite.complete_bipartite_graph(5, 5)


# Dibuja el grafo
pos = nx.spring_layout(g, seed=42)
nx.draw(g, pos=pos, with_labels=True, node_color='skyblue',
        node_size=700, font_weight='bold', edge_color='gray')

# Muestra el grafo
plt.show()


G = nx.adjacency_matrix(g).toarray()
print(G)


def get_all_paths(G, start_vertex, current_vertex, current_path, visited_paths):
    """
    Función que retorna todos los caminos posibles en un grafo a partir de cierto vértice.
    Es poco eficiente.
    """
    for i in range(len(G)):
        if (G[current_vertex][i] == 1) and (i not in current_path) and (tuple(current_path) not in visited_paths.keys()):
            current_path.append(i)
            get_all_paths(G, start_vertex, i, current_path, visited_paths)
            visited_paths[tuple(current_path)] = True
            current_path.pop()
    return visited_paths


def has_odd_cycles_recursive(G, start_vertex, current_vertex, current_path, visited_paths):
    """
    Función que detecta si hay ciclos impares.
    Esta función es poco eficiente dado que analiza todos los posibles caminos.
    """
    for i in range(len(G)):
        if (G[current_vertex][i] == 1) and (i not in current_path) and (tuple(current_path) not in visited_paths.keys()):
            current_path.append(i)
            has_odd_cycles_recursive(G, start_vertex, i, current_path, visited_paths)
            if (G[current_path[-1]][start_vertex] == 1) and (len(current_path) > 2):
                if (len(current_path) + 1) % 2 == 0:
                    print(current_path)
                    return True
            visited_paths[tuple(current_path)] = True
            current_path.pop()
    return False


def is_bipartite_dfs(G, current_vertex, prev_vertex, visited, vertex_colors):
    """
    Función que detecta si grafo es bipartito tratando de 2-colorear mediante DFS.
    """
    visited[current_vertex] = True

    if (current_vertex not in vertex_colors) and (prev_vertex in vertex_colors):
        nw_colors = []
        for i in range(len(G)):
            if (G[current_vertex][i] == 1):
                if i in vertex_colors:
                    nw_colors.append(vertex_colors[i])
        nw_colors = list(set(nw_colors))
        print(nw_colors)
        if len(nw_colors) == 2:
            return False
        elif nw_colors[0] == "red":
            vertex_colors[current_vertex] = 'yellow'
        elif nw_colors[0] == 'yellow':
            vertex_colors[current_vertex] = 'red'

    elif (prev_vertex not in vertex_colors):
        vertex_colors[current_vertex] = 'red'

    for i in range(len(G)):
        if (G[current_vertex][i] == 1) and not (visited[i]):
            if not is_bipartite_dfs(G, i, current_vertex, visited, vertex_colors):
                return False

    return True


a = get_all_paths(G, 0, 0, [0], {})
b = has_odd_cycles_recursive(G, 0, 0, [0], {})


c = is_bipartite_dfs(G, 0, -1, [False] * len(G), {})
print(b)
