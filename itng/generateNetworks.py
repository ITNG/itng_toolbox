import os
import bct
import igraph
import numpy as np
import pylab as pl
from sys import exit
from copy import copy
import networkx as nx
from random import shuffle
import matplotlib.pyplot as plt
# pl.switch_backend('agg')


class graphGenerator:
    ''' 
    generate graphs.
    '''

    def __init__(self, seed=None):

        self.G = 0
        if seed:
            self.seed = seed
            np.random.seed(seed)
        else:
            self.seed = None

    # ---------------------------------------------------------------#
    def print_adjacency_matrix(self, G):
        '''
        print adjacency matrix of the graph on the screen 
        '''
        M = nx.to_numpy_matrix(G)
        M = np.array(M)
        for row in M:
            for val in row:
                print('%.0f' % val),
            print()

    # ---------------------------------------------------------------#
    def complete_graph(self, N, plot_adj=False, **kwargs):
        ''' 
        Return the complete graph with N nodes.

        :param N: number of nodes
        :param plot_adj: to plot the adjacency matrix
        :param kwargs: other argument for ploting the adjacency matrix
        :return: [ndarray] adjacency matrix 
        '''

        self.N = N
        self.G = nx.complete_graph(N)
        M = nx.to_numpy_array(self.G)

        if plot_adj:
            self.imshow_plot(M, **kwargs)

        return M
    # ---------------------------------------------------------------#

#     def complete_weighted(self, N, clusters, weights, d, plot_adj=False):
#         '''
#         return a complete weighted graph, weights distribute in clusters
#         with the same size.

#         :param N: number of nodes.
#         :param weights: list with length 2, shows the weights of edges in and between clusters, respectivly
#         :param clusters: list of cluster lengths
#         :param d: delay for all nodes
#         :return: adjacency matrix and delay matrix
#         '''

#         self.N = N
#         # I use delay as weight
#         self.modular_graph(N, 1, 1, clusters, weights[0], weights[1])
#         M = self.D
#         if plot_adj:
#             self.imshow_plot(np.asarray(M).reshape(N, N), "con")

#         D = self.complete_graph(N) * d
#         return M, D
#     # ---------------------------------------------------------------#

#     def complete_hmn_weighted(self, n_M0, level, weights, delay,
#                               plot_adj=False, seed=124):
#         '''
#         return a complete weighted graph, weights distribute in a hierarchical
#         modular form.
#         Single delay for every node.
#         '''
#         self.hierarchical_modular_graph(n_M0, level, 1, 1, 1, weights,
#                                         plot_adj=False, seed=124)
#         N = self.N
#         M = nx.to_numpy_matrix(self.G, weight='weight')

#         if plot_adj:
#             self.imshow_plot(M, "con")
#         self.D = (delay,)*N*N

#         return tuple(np.asarray(M).reshape(-1))

    # ---------------------------------------------------------------#
    def gnp_random_graph(self, n, p, plot_adj=False, **kwargs):
        ''' 
        Returns Erdos Renyi graph G(n,p)

        :param n: [int] number of nodes
        :param p: [float] probability of existing an edge
        :param plot_adj: [bool, optional(default=False)] if True plot the adjacency matrix
        :return: [ndarray] adjacency matrix 
        '''

        G = nx.gnp_random_graph(n, p, seed=self.seed, directed=False)
        M = nx.to_numpy_matrix(G)

        if plot_adj:
            self.imshow_plot(M, **kwargs)

        return M
    # ---------------------------------------------------------------#

    def gnm_random_graph(self, n, m, plot_adj=False, **kwargs):
        ''' 
        Returns Erdos Renyi graph G(n,m)

        :param n: [int] number of nodes
        :param m: [int] number of edges
        :param plot_adj: [bool, optional(default=False)] if True plot the adjacency matrix
        :return: [ndarray] adjacency matrix 
        '''

        G = nx.gnm_random_graph(n, m, seed=self.seed, directed=False)
        M = nx.to_numpy_matrix(G)

        if plot_adj:
            self.imshow_plot(M, **kwargs)

        return M
    # ---------------------------------------------------------------#

    def modular_graph(self, N, pIn, pOut, lengths, dIn, dOut, plot_adj=False):
        ''' returns a modular networks

        :param N: number of nodes
        :param pIn: conectivity of nodes insede clusters
        :param pOut: conectivity of nodes between clusters
        :param n_cluster:  number of clusters in graph
        :param dIn: delay between nodes inside the clusters
        :param dOut: delay between nodes outside the clusters
        :param plot_adj: [bool, optional(default=False)] if True plot the adjacency matrix
        :return: [ndarray] adjacency matrix 
        '''

        M = np.zeros((N, N))
        D = np.zeros((N, N))
        n_cluster = len(lengths)

        for i in range(N):
            for j in range(i+1, N):
                r = np.random.rand()
                if r < pOut:
                    M[i, j] = M[j, i] = 1.0
                    D[i, j] = D[j, i] = dOut

        # empty inside the clusters
        s = 0
        for k in range(n_cluster):
            if k > 0:
                s += lengths[k-1]
            for i in range(s, (lengths[k]+s)):
                for j in range(i+1, (lengths[k]+s)):
                    M[i, j] = M[j, i] = 0.0
                    D[i, j] = D[j, i] = 0.0

        # fill inside the clusters
        s = 0
        for k in range(n_cluster):
            if k > 0:
                s += lengths[k-1]
            for i in range(s, (lengths[k]+s)):
                for j in range(i+1, (lengths[k]+s)):
                    r = np.random.rand()
                    if r < pIn[k]:
                        M[i, j] = M[j, i] = 1.0
                        D[i, j] = D[j, i] = dIn

        # print delay matrix
        def print_delay_matrix():
            ofi = open("delay.txt", "w")
            for i in range(N):
                for j in range(N):
                    ofi.write("%2.0f" % D[i, j])
                ofi.write("\n")
            ofi.close()

        self.G = nx.from_numpy_matrix(M)
        self.D = D

        if plot_adj:
            self.imshow_plot(M, "con")

        return M
    # ---------------------------------------------------------------#

    def modular_graph_fixed_num_edges(self, N,
                                      n_edges_in,
                                      n_edges_between,
                                      clusters,
                                      delayIn,
                                      delayOut,
                                      plot_adj=False):

        """
        Returns a modular graph with fixed number of edges

        :param N: [int] number of nodes
        :param n_edges_in: [int] number of edges inside the clusters
        :param n_edges_between: [int] number of edges between clusters
        :param clusters: [list of int] size of each module 
        :param delayIn: [float] delay of edges inside the modules
        :param delayOut: [float] delay of edges between the modules
        :param plot_adj: [bool, optional(default=False)] if True plot the adjacency matrix
        :return: [ndarray] adjacency matrix 
        """



        n_clusters = len(clusters)
        probs = np.asarray(clusters)/float(N)
        n_edges_in_arr = [int(np.rint(i * n_edges_in)) for i in probs]

        a = [0]
        nodes = []
        for i in range(n_clusters):
            a.append(a[i]+clusters[i])
            nodes.append(range(a[i], a[i+1]))

        M = []
        labels = np.cumsum(clusters[:-1]).tolist()
        labels.insert(0, 0)
        G = nx.Graph()

        for i in range(n_clusters):
            M0 = nx.gnm_random_graph(clusters[i], n_edges_in_arr[i],
                                     seed=self.seed+i)
            tG = nx.Graph()
            for e in nx.edges(M0):
                tG.add_edge(*e, delay=delayIn)
            tG = nx.convert_node_labels_to_integers(tG, labels[i])
            G = nx.compose(G, tG)

        counter = 0
        while counter != n_edges_between:
            e = np.random.randint(0, N, size=2)
            if e[0] == e[1]:
                continue
            else:
                condition = check_edge_between_clusters(e, nodes)
                if condition & (~G.has_edge(*e)):
                    G.add_edge(*e, delay=delayOut)
                    counter += 1

        # Adj = nx.to_numpy_array(G)
        # D = extract_attributes(G, 'delay')

        # density_clusters_info(G, clusters)
        # print nx.info(G)

        return G

    # ---------------------------------------------------------------#

#     def hierarchical_modular_graph(self, n_M0, level, prob0, prob, alpha,
#                                    delays, plot_adj=False):
#         '''
#         n_M0 : size of module at level 1
#         s    : number of levels
#         n_modules : number of modules
#         N : number of nodes
#         ps : probability of conection in each level
#              level one is 1 and the others determine with prob function
#         delays: delay in each level as a list
#         '''
#         def probability(l, a=1, p=0.25):
#             if l == 0:
#                 from sys import exit
#                 print("level shold be integer and > 0", l)
#                 exit(0)
#             else:
#                 return a * p**l

#         s = level
#         n_modules = int(2**(s-1))  # number of modules
#         N = int(n_modules*n_M0)  # number of nodes
#         self.N = N

#         # M0 = nx.complete_graph(n_M0)
#         M0 = nx.erdos_renyi_graph(n_M0, prob0, seed=self.seed)
#         for e in nx.edges(M0):
#             M0.add_edge(*e, weight=delays[0])  # delays act in weight attribute

#         ps = [prob0]+[probability(i, alpha, prob) for i in range(1, s)]

#         for l in range(1, s):
#             if l == 1:
#                 M_pre = [M0] * n_modules
#             else:
#                 M_pre = copy(M_next)
#             M_next = []
#             k = 0
#             for ii in range(n_modules/(2**l)):
#                 step = 2**(l-1)
#                 tG = nx.convert_node_labels_to_integers(M_pre[k+1], step*n_M0)
#                 tG1 = nx.compose(M_pre[k], tG)
#                 edge = 0
#                 effort = 1
#                 ''' make sure that connected modules are not isolated '''
#                 while edge < 1:
#                     # print "effort ", effort
#                     effort += 1
#                     for i in range(len(tG1)):
#                         for j in range(i+1, len(tG1)):
#                             if (i < step*n_M0) & (j > step*n_M0-1) & (np.random.rand() < ps[l]):
#                                 tG1.add_edge(i, j, weight=delays[l])
#                                 edge += 1

#                 M_next.append(tG1)
#                 k += 2
#         self.G = M_next[0]

#         if plot_adj:
#             M = nx.to_numpy_matrix(self.G, weight=None)
#             self.imshow_plot(M, "con")
#             # self.print_adjacency_matrix(self.G)
#         D = nx.to_numpy_matrix(self.G, weight='weight')
#         self.D = np.asarray(D)

#         M = nx.to_numpy_matrix(self.G, weight=None)

#         return np.asarray(M)
#     # ---------------------------------------------------------------#

#     def from_adjacency_matrix_graph(self, filename, plot_adj=False):
#         ''' makes a graph from adjacency matrix in filename
#         and return 1D double vector in stl '''

#         A = np.genfromtxt(filename)
#         self.N = len(A)

#         if plot_adj:
#             self.imshow_plot(A, "con")

#         return A
#     # ---------------------------------------------------------------#

#     def imshow_plot(self, data, name, title=None):
#         from mpl_toolkits.axes_grid1 import make_axes_locatable
#         fig = pl.figure(140, figsize=(6, 6))
#         pl.clf()
#         ax = pl.subplot(111)
#         im = ax.imshow(data, interpolation='nearest',
#                        cmap='afmhot')  # , cmap=pl.cm.ocean
#         # ax.invert_yaxis()
#         if title:
#             ax.set_title(title)
#         divider = make_axes_locatable(ax)
#         cax = divider.append_axes("right", size="5%", pad=0.05)
#         pl.colorbar(im, cax=cax)
#         pl.savefig('../data/fig/'+name+".png", dpi=300)
#         # pl.close()
#     # ---------------------------------------------------------------#

#     def multilevel(self, data):
#         conn_indices = np.where(data)
#         weights = data[conn_indices]
#         edges = zip(*conn_indices)
#         G = igraph.Graph(edges=edges, directed=False)
#         G.es['weight'] = weights
#         comm = G.community_multilevel(weights=weights, return_levels=False)
#         return comm
#     # ---------------------------------------------------------------#

#     def hmn_cortex_alike(self, len_communities=None,
#                          pIn_clusters=None, p2=0.2, p3=0.04,
#                          weight1=1.0, weight2=0.5, weight3=0.2,
#                          delay1=5.0, delay2=20.0, delay3=25.0,
#                          plot_adj=False):

#         if len_communities is None:
#             len_communities = [12, 13, 14, 14, 12]
#             pIn_clusters = [0.515, 0.731, 0.670, 0.527, 0.712]

#         number_of_communities = len(len_communities)
#         assert (number_of_communities == len(pIn_clusters))
#         N = np.sum(len_communities)

#         # ---------------------------------------------------------------#
#         def fill_in_modules():

#             G = nx.Graph()
#             M = []
#             for i in range(number_of_communities):
#                 tg = nx.erdos_renyi_graph(len_communities[i],
#                                           pIn_clusters[i], seed=self.seed)
#                 M.append(tg)
#                 M[i] = nx.convert_node_labels_to_integers(M[i], nodes[i][0])
#                 G = nx.compose(G, M[i])
#             return G.edges()
#         # ---------------------------------------------------------------#

#         def fill_between_modules(p2, p3):

#             def empty_inside_modules(M):
#                 s = 0
#                 for k in range(number_of_communities):
#                     if k > 0:
#                         s += len_communities[k-1]
#                     for i in range(s, (len_communities[k]+s)):
#                         for j in range(i+1, (len_communities[k]+s)):
#                             M[i, j] = M[j, i] = 0.0
#                 return M

#             M = np.zeros((N, N))

#             for i in range(N/2):
#                 for j in range(i+1, N/2):
#                     r = np.random.rand()
#                     if r < p2:
#                         M[i, j] = M[j, i] = 1.0

#             for i in range(N/2, N):
#                 for j in range(i+1, N):
#                     r = np.random.rand()
#                     if r < p2:
#                         M[i, j] = M[j, i] = 1.0

#             M = empty_inside_modules(M)
#             tg2 = nx.from_numpy_matrix(M)
#             edges2 = tg2.edges()

#             M = np.zeros((N, N))
#             for i in range(N):
#                 for j in range(i+1, N):
#                     r = np.random.rand()
#                     if r < p3:
#                         M[i, j] = M[j, i] = 1.0

#             M = empty_inside_modules(M)
#             tg3 = nx.from_numpy_matrix(M)
#             edges3 = tg3.edges()

#             return edges2, edges3
#         # ---------------------------------------------------------------#
#         # pOut = 0.157

#         nodes = []
#         a = [0]
#         for i in range(number_of_communities):
#             a.append(a[i]+len_communities[i])
#             nodes.append(range(a[i], a[i+1]))

#         edges1 = fill_in_modules()
#         edges2, edges3 = fill_between_modules(p2, p3)

#         G = nx.Graph()
#         for e in edges1:
#             G.add_edge(*e, weight=weight1, delay=delay1)
#         for e in edges2:
#             G.add_edge(*e, weight=weight2, delay=delay2)
#         for e in edges3:
#             G.add_edge(*e, weight=weight3, delay=delay3)

#         if plot_adj:
#             A = extract_attributes(G, "weight")
#             D = extract_attributes(G, "delay")
#             self.imshow_plot(A, "A", title="Coupling weights")
#             self.imshow_plot(D, "D", title='Delays')

#         print(nx.info(G))

#         return A, D

# # ---------------------------------------------------------------------- #


# def rewiring_modular_graph(G,
#                            clusters,
#                            delay,
#                            ens=1,
#                            step_iteration=100,
#                            threshold=0.185,
#                            plot_adj=False):
#     """
#     threshold : threshold of modularity, determine hot many
#                 edges should be removed from clusters.
#     """

#     # density_clusters_info(G, clusters)

#     ofile = open("../data/text/lambda-"+str(ens)+".txt", "w")
#     ffile = open("../data/text/fraction.txt", "a")

#     if plot_adj:
#         imshow_plot(nx.to_numpy_array(G),
#                         "../data/fig/before-"+str(ens)+".png")

#     print("lambda2/lambdan : %g" % lambdan_over_lambda2(G))

#     n_nodes = nx.number_of_nodes(G)
#     n_clusters = len(clusters)

#     edges = nx.edges(G)
#     for e in edges:
#         G.add_edge(*e, delay=delay)

#     n_edges = len(edges)
#     edges = list(edges)

#     a = [0]
#     nodes = []
#     for i in range(n_clusters):
#         a.append(a[i]+clusters[i])
#         nodes.append(range(a[i], a[i+1]))

#     # put edges in clusters into a list
#     edges_in = []
#     for i in range(n_clusters):
#         for e in edges:
#             if ((e[0] in nodes[i]) & (e[1] in nodes[i])):
#                 edges_in.append(e)
#     shuffle(edges_in)
#     n_edges_in = len(edges_in)

#     counter = 0
#     for it in range(n_edges_in):
#         ei = np.random.randint(0, n_edges_in)
#         edge = edges_in[ei]
#         # print G.has_edge(*edge)
#         G.remove_edge(*edge)
#         del edges_in[ei]
#         n_edges_in -= 1

#         condition = False
#         while True:
#             e = np.random.randint(0, n_nodes, size=2)
#             # print e
#             if e[0] == e[1]:
#                 continue
#             else:
#                 condition = check_edge_between_clusters(e, nodes)
#                 if condition & (~G.has_edge(*e)):
#                     G.add_edge(*e, delay=delay)
#                     break
#         if (it % 5) == 0:
#             adj = nx.to_numpy_array(G)
#             modularity = bct.modularity_und(adj)[1]
#             frac = lambdan_over_lambda2(G)

#             ofile.write("%6d %12.6f %12.6f\n" %
#                         (it, frac, modularity))
#             if ((it % step_iteration) == 0):
#                 Adj = nx.to_numpy_array(G)
#                 D = extract_attributes(G, "delay")
#                 np.savetxt(str("../data/text/C-%d-%d.txt" % (counter, ens)),
#                            Adj, fmt='%d')
#                 np.savetxt(str("../data/text/D-%d-%d.txt" % (counter, ens)),
#                            D, fmt='%d')
#                 counter += 1
#                 ffile.write("%10.3f" % frac)

#             if modularity < threshold:
#                 break
#     ffile.write("\n")
#     ffile.close()

#     ofile.close()
#     if plot_adj:
#         imshow_plot(nx.to_numpy_array(
#             G), "../data/fig/after-"+str(ens)+".png")

#     # density_clusters_info(G, clusters)
# # ---------------------------------------------------------------------- #


# def check_edge_between_clusters(e, nodes):
#     "check if given edge is between clusters."
#     # print e[0], e[1]
#     n_clusters = len(nodes)
#     s = range(n_clusters)
#     shuffle(s)
#     for i in s:
#         if ((e[0] in nodes[i]) & (e[1] not in nodes[i])) |\
#                 ((e[0] not in nodes[i]) & (e[1] in nodes[i])):
#             return True

#     return False

# # ---------------------------------------------------------------------- #


# def extract_attributes(G, attr=None):
#     edges = G.edges()
#     n = nx.number_of_nodes(G)
#     A = np.zeros((n, n))

#     for e in edges:
#         A[e[0], e[1]] = A[e[1], e[0]] = G[e[0]][e[1]][attr]
#     return A
# # ---------------------------------------------------------------------- #


# def density_clusters_info(G, clusters, print_result=True):

#     N = nx.number_of_nodes(G)
#     n_clusters = len(clusters)

#     edges = G.edges()
#     num_edges = nx.number_of_edges(G)
#     num_edges_in = 0

#     a = [0]
#     nodes = []
#     for i in range(n_clusters):
#         a.append(a[i]+clusters[i])
#         nodes.append(range(a[i], a[i+1]))

#     if print_result:
#         print("="*70)
#         print('%s%13s%15s%15s%15s' % (
#             'index', 'size', 'density',
#             'n_edges_in', 'n_edges_out'))

#     num_edges_in_total = 0
#     num_edges_out_total = 0
#     for i in range(n_clusters):
#         num_e_in = 0
#         num_e_out = 0
#         for e in edges:
#             if ((e[0] in nodes[i]) & (e[1] in nodes[i])):
#                 num_e_in += 1
#             elif ((e[0] in nodes[i]) & (e[1] not in nodes[i])) |\
#                     ((e[0] not in nodes[i]) & (e[1] in nodes[i])):
#                 num_e_out += 1

#         num_edges_in_total += num_e_in
#         num_edges_out_total += num_e_out
#         density_in = 2.0 * num_e_in / float(clusters[i]*(clusters[i]-1))

#         if print_result:
#             print('%1d%15d%15.2f%15d%15d' % (i, clusters[i],
#                                              density_in, num_e_in, num_e_out))
#     edges_between = num_edges-num_edges_in_total
#     if print_result:
#         print("number of edges : %d" % len(edges))
#         print("number of edges in clusters : %d" % num_edges_in_total)
#         print("number of edges between clusters : %d" %
#               (num_edges_out_total/2))
#         print("fraction of in links : %g " % (
#             num_edges_in_total / float(num_edges)))
#         print("fraction of between links : %g " % (
#             edges_between/float(num_edges)))

#     return (num_edges_in_total, edges_between)
# # --------------------------------------------------------------#


# def lambdan_over_lambda2(G):

#     L = nx.laplacian_matrix(G)
#     eig = np.linalg.eigvals(L.A)
#     eig = np.sort(eig)
#     r = eig[-1]/float(eig[1])

#     return r

# # --------------------------------------------------------------#


# def imshow_plot(data, fname='R', cmap='afmhot',
#                 figsize=(5, 5),
#                 vmax=None, vmin=None):
#     from mpl_toolkits.axes_grid1 import make_axes_locatable
#     fig = pl.figure(100, figsize=figsize)
#     pl.clf()
#     ax = pl.subplot(111)
#     im = ax.imshow(data, interpolation='nearest', cmap=cmap,
#                    vmax=vmax, vmin=vmin)
#     ax.invert_yaxis()
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="5%", pad=0.05)
#     pl.colorbar(im, cax=cax)
#     pl.savefig(fname, dpi=150)
#     pl.close()
