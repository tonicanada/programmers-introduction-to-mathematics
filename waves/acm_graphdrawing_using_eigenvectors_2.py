"""
Este programa realiza las siguientes tareas:

1. Genera un grafo aleatorio con n vértices.
2. Plotea el grafo con una disposición aleatoria.
3. Calcula los eigenvectores y eigenvalores del laplaciano del grafo.
4. Plotea el grafo utilizando el 2º y 3º eigenvector del laplaciano para determinar la posición de los nodos.

El propósito de este programa es demostrar cómo la estructura espectral del laplaciano de un grafo 
(específicamente, sus eigenvectores) puede ser utilizada para visualizar el grafo de una manera que refleje 
sus propiedades estructurales.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Genera un grafo aleatorio con n vértices
n = 10  # Puedes cambiar este valor
G = nx.gnp_random_graph(n, 0.5)  # 0.5 es la probabilidad de que exista una arista entre dos nodos

# # Dibuja el grafo aleatorio
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# nx.draw(G, with_labels=True, node_color='lightblue')
# plt.title('Grafo Aleatorio')

# # Calcula el laplaciano del grafo
# L = nx.laplacian_matrix(G).todense()

# # Calcula eigenvalores y eigenvectores del laplaciano
# eigenvalues, eigenvectors = np.linalg.eigh(L)

# # Ordena los eigenvectores basados en los eigenvalores
# sorted_indices = np.argsort(eigenvalues)
# eigenvectors = eigenvectors[:, sorted_indices]

# # Usa el 2º y 3º eigenvector para las coordenadas
# x_coords = eigenvectors[:, 1]
# y_coords = eigenvectors[:, 2]

# # Dibuja el grafo usando las coordenadas de los eigenvectores del laplaciano
# plt.subplot(1, 2, 2)
# nx.draw(G, with_labels=True, pos=dict(zip(range(n), zip(x_coords, y_coords))), node_color='lightgreen')
# plt.title('Grafo usando 2º y 3º Eigenvectores del Laplaciano')

# plt.tight_layout()
# plt.show()



import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay


# Genera un grafo aleatorio con n vértices
n = 7  # Puedes cambiar este valor
# G = nx.gnp_random_graph(n, 0.5)  # 0.5 es la probabilidad de que exista una arista entre dos nodos
adj_matrix = np.array([[0,1,1,0],
                       [1,0,0,1],
                       [1,0,0,1],
                       [0,1,1,0]])

adj_matrix = np.array([[0,1,1,0,0],
                       [1,0,1,1,1],
                       [1,1,0,1,0],
                       [0,1,1,0,1],
                       [0,1,0,1,0]])

A = np.zeros((12, 12))
A += np.diag(np.ones(11), 1)
A += A.T
A[0, 11] = 1
A[11, 0] = 1

# # Example 4
# A = np.zeros((20, 20))
# xy = np.random.rand(20, 2)
# trigs = Delaunay(xy).simplices
# elemtrig = np.ones((3, 3)) - np.eye(3)
# for trig in trigs:
#     A[trig[:, None], trig] = elemtrig
# A = (A > 0).astype(int)


print(A)

adj_matrix = np.array(A)
print(A)

G = nx.from_numpy_array(adj_matrix)
G = nx.dodecahedral_graph()

# Configurar el tamaño de la figura
plt.figure(figsize=(12, 6))

# Dibujar el grafo aleatorio a la izquierda
plt.subplot(1, 2, 1)
nx.draw(G, with_labels=True, node_color='lightblue')
plt.title('Grafo Aleatorio')

# Calcular la matriz laplaciana
L = nx.laplacian_matrix(G).toarray()

# Encontrar eigenvectores y eigenvalores
eigenvalues, eigenvectors = np.linalg.eigh(L)

# Tomar los eigenvectores correspondientes al segundo y tercer eigenvalor más pequeño
x_coords = eigenvectors[:, 1]
y_coords = eigenvectors[:, 2]

# Crear un diccionario de posiciones usando una lista ordenada de nodos
nodes_list = sorted(G.nodes())
pos = {node: (x_coords[i], y_coords[i]) for i, node in enumerate(nodes_list)}

# Dibujar el grafo basado en los eigenvectores a la derecha
plt.subplot(1, 2, 2)
nx.draw(G, pos, with_labels=True, node_color='lightgreen')
plt.title('Grafo Basado en Eigenvectores')

plt.show()

