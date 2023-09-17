import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Suponiendo que A es tu matriz de adyacencia
# adj_matrix = np.array(A)
G = nx.dodecahedral_graph()


# Calcular la matriz laplaciana
L = nx.laplacian_matrix(G).toarray()

# Encontrar eigenvectores y eigenvalores
eigenvalues, eigenvectors = np.linalg.eigh(L)

# Tomar los eigenvectores correspondientes al segundo, tercer y cuarto eigenvalor más pequeño
x_coords = eigenvectors[:, 1]
y_coords = eigenvectors[:, 2]
z_coords = eigenvectors[:, 3]

# Crear un diccionario de posiciones usando una lista ordenada de nodos
nodes_list = sorted(G.nodes())
pos = {node: (x_coords[i], y_coords[i], z_coords[i]) for i, node in enumerate(nodes_list)}

# Configurar el tamaño de la figura
fig = plt.figure(figsize=(12, 6))

# Dibujar el grafo en 3D basado en los eigenvectores
ax = fig.add_subplot(111, projection='3d')
for edge in G.edges():
    x = np.array((pos[edge[0]][0], pos[edge[1]][0]))
    y = np.array((pos[edge[0]][1], pos[edge[1]][1]))
    z = np.array((pos[edge[0]][2], pos[edge[1]][2]))
    ax.plot(x, y, z, color='k')

ax.scatter(x_coords, y_coords, z_coords, c='lightgreen', s=100)
for i, node in enumerate(nodes_list):
    ax.text(x_coords[i], y_coords[i], z_coords[i], s=str(node))

ax.set_title('Grafo 3D Basado en Eigenvectores')
plt.show()
