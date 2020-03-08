import itng
from itng import networks
import networkx as nx
from itng.drawing import Drawing


g = itng.networks.Generators()
G = g.complete_graph(10)
Drawing.plot_adjacency_matrix(G,
                              fileName="images/complete_graph.png",
                              cmap="jet")


G = g.modular_graph(30, 0.7, 0.2, [15] * 2, 2.0, 4.5)
Drawing.plot_adjacency_matrix(G, fileName="images/modular.png",
                              cmap="jet")


# G = g.modular_network_fixed_num_edges(50, 100, 50, [10] * 5, 0.1, 0.6, verbosity=False)

G = g.hierarchical_modular_graph(20, 3, 0.8, 0.25, 1.0, [2.0, 4.5, 6.1])
Drawing.plot_adjacency_matrix(G, fileName="images/hmn.png", cmap="jet")
print (nx.info(G))
