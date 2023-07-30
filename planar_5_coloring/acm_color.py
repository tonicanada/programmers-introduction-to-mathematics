import igraph as ig
import color

G = ig.Graph(n=8)
labels = list(range(8))
G.add_edges([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
            (4, 2), (0, 2), (3, 6), (6, 1), (1, 5), (6, 5), (7, 6), (7, 3), (1, 3)])



layout = G.layout("fruchterman_reingold")

ig.plot(G, vertex_label=labels, vertex_label_dist=2, bbox=(500, 500), layout=layout)

# print(G)

G2 = color.planar_five_color(G)

layout = G2.layout("fruchterman_reingold")

ig.plot(G2, vertex_label=G2.vs['color'], vertex_label_dist=2, layout=layout)
