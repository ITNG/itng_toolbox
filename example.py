import itng

g = itng.graphGenerator()
# A, W = g.complete_graph(10, plot_adj=True, weight=2)
# A, W = g.gnp_random_graph(10, 0.5, 2.0, True)
A, W = g.modular_graph(10, 0.7, 0.1, [5] * 2, 1, 2, True)
print(A)
print(W)


