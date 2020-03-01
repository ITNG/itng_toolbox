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


class networkGenerator:
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
        M = nx.to_numpy_array(G)
        M = np.array(M)
        for row in M:
            for val in row:
                print('%.0f' % val),
            print()

    # ---------------------------------------------------------------#

    def extract_attributes(self, G, attr=None):
        """
        extract the matrix of given atributes from graph

        :param G: [networkx graph object]
        :param attr: [string] given attribute
        :return: [ndarray] matrix of given attribute
        """
        edges = G.edges()
        n = nx.number_of_nodes(G)
        A = np.zeros((n, n))

        for e in edges:
            A[e[0], e[1]] = A[e[1], e[0]] = G[e[0]][e[1]][attr]
        return A
    # ---------------------------------------------------------------#

    def plot_adjacency_matrix(self, data,
                              fileName='R',
                              cmap='afmhot',
                              figsize=(5, 5),
                              labelsize=None,
                              xyticks=True,
                              vmax=None,
                              vmin=None,
                              ax=None):
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        saveFig = False

        if ax is None:
            fig, ax = pl.subplots(1, figsize=figsize)
            saveFig = True

        im = ax.imshow(data, interpolation='nearest',
                       cmap=cmap, vmax=vmax,
                       vmin=vmin, origin="lower")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = pl.colorbar(im, cax=cax)

        if labelsize:
            cbar.ax.tick_params(labelsize=labelsize)

        if xyticks == False:
            ax.set_xticks([])
            ax.set_yticks([])

        if saveFig:
            pl.savefig(fileName)
            pl.close()
    # ---------------------------------------------------------------#

    def complete_network(self, N,
                         plot_adj=False,
                         **kwargs):
        ''' 
        Return the complete graph with N nodes.

        :param N: number of nodes        
        :param plot_adj: to plot the adjacency matrix
        :param kwargs: other argument for ploting the adjacency matrix
        :return: [ndarray] adjacency matrix
        '''

        self.N = N
        self.G = nx.complete_graph(N)
        adjMatrix = nx.to_numpy_array(self.G, dtype=int)

        if plot_adj:
            self.plot_adjacency_matrix(adjMatrix, **kwargs)

        return adjMatrix
    # ---------------------------------------------------------------#

    # def complete_weighted_graph(self, N, clusters, weights, d, plot_adj=False):
    #     '''
    #     return a complete weighted graph, weights distribute in clusters
    #     with the same size.

    #     :param N: number of nodes.
    #     :param weights: list with length 2, shows the weights of edges in and between clusters, respectivly
    #     :param clusters: list of cluster lengths
    #     :param d: delay for all nodes
    #     :return: adjacency matrix and delay matrix
    #     '''

    #     self.N = N
    #     # I use delay as weight
    #     self.modular_graph(N, 1, 1, clusters, weights[0], weights[1])
    #     M = self.D
    #     if plot_adj:
    #         self.plotAdjacencyMatrix(np.asarray(M).reshape(N, N), "con")

    #     D = self.complete_graph(N) * d
    #     return M, D
    # ---------------------------------------------------------------#

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
#         M = nx.to_numpy_array(self.G, weight='weight')

#         if plot_adj:
#             self.plotAdjacencyMatrix(M, "con")
#         self.D = (delay,)*N*N

#         return tuple(np.asarray(M).reshape(-1))

    # ---------------------------------------------------------------#
    def gnp_random_network(self,
                           n,
                           p,
                           plot_adj=False,
                           **kwargs):
        ''' 
        Returns Erdos Renyi network, given size and probability 

        :param n: [int] number of nodes
        :param p: [float] probability of existing an edge
        :param plot_adj: [bool, optional(default=False)] if True plot the adjacency matrix
        :return: [ndarray] adjacency matrix 
        '''

        G = nx.gnp_random_graph(n, p, seed=self.seed, directed=False)
        adjMatrix = nx.to_numpy_array(G, dtype=int)

        if plot_adj:
            self.plot_adjacency_matrix(adjMatrix, **kwargs)

        return adjMatrix
    # ---------------------------------------------------------------#

    def gnm_random_network(self, n, m, plot_adj=False, **kwargs):
        ''' 
        Returns Erdos Renyi network, given size and number of edges

        :param n: [int] number of nodes
        :param m: [int] number of edges
        :param plot_adj: [bool, optional(default=False)] if True plot the adjacency matrix
        :return: [ndarray] adjacency matrix 
        '''

        G = nx.gnm_random_graph(n, m, seed=self.seed, directed=False)
        adjMatrix = nx.to_numpy_array(G, dtype=int)

        if plot_adj:
            self.plot_adjacency_matrix(adjMatrix, **kwargs)

        return adjMatrix
    # ---------------------------------------------------------------#

    def modular_network(self, N,
                        pIn,
                        pOut,
                        sizes,
                        wIn,
                        wOut,
                        plot_adj=False,
                        **kwargs):
        ''' 
        returns a modular networks

        :param N: [int] number of nodes
        :param pIn: [int, float or list of numbers] probability of existing an edge inside clusters
        :param pOut: [int, float] probability of existing an edge between clusters
        :param sizes: [list of int] size of clusters in network
        :param wIn: weight of edges inside clusters
        :param wOut: weight of edges between clusters
        :param plot_adj: [bool, optional(default=False)] if True plot the adjacency matrix
        :return: [2 ndarray] adjacency matrix and weighted matrix
        '''

        num_cluster = len(sizes)

        if isinstance(pIn, list) == False:
            pIn = [pIn] * num_cluster

        A = np.zeros((N, N))
        W = np.zeros((N, N))

        for i in range(N):
            for j in range(i+1, N):
                r = np.random.rand()
                if r < pOut:
                    A[i, j] = A[j, i] = 1.0
                    W[i, j] = W[j, i] = wOut

        # empty inside the clusters
        s = 0
        for k in range(num_cluster):
            if k > 0:
                s += sizes[k-1]
            for i in range(s, (sizes[k]+s)):
                for j in range(i+1, (sizes[k]+s)):
                    A[i, j] = A[j, i] = 0.0
                    W[i, j] = W[j, i] = 0.0

        # fill inside the clusters
        s = 0
        for k in range(num_cluster):
            if k > 0:
                s += sizes[k-1]
            for i in range(s, (sizes[k]+s)):
                for j in range(i+1, (sizes[k]+s)):
                    r = np.random.rand()
                    if r < pIn[k]:
                        A[i, j] = A[j, i] = 1.0
                        W[i, j] = W[j, i] = wIn

        if plot_adj:
            self.plot_adjacency_matrix(A, **kwargs)

        return A, W
    # ---------------------------------------------------------------#

    def modular_network_fixed_num_edges(self, N,
                                        n_edges_in,
                                        n_edges_between,
                                        sizes,
                                        wIn,
                                        wOut,
                                        plot_adj=False):
        """
        Returns a modular network with fixed number of edges

        :param N: [int] number of nodes
        :param n_edges_in: [int] number of edges inside the clusters
        :param n_edges_between: [int] number of edges between clusters
        :param sizes: [list of int] size of each module 
        :param wIn: [float] weight of edges inside the clusters
        :param wOut: [float] weight of edges between the clusters
        :param plot_adj: [bool, optional(default=False)] if True plot the adjacency matrix
        :return: [2 ndarray] adjacency matrix and weighted matrix
        """

        num_clusters = len(sizes)
        probs = np.asarray(sizes)/float(N)
        n_edges_in_arr = [int(np.rint(i * n_edges_in)) for i in probs]

        tmp = [0]
        nodes = []
        for i in range(num_clusters):
            tmp.append(tmp[i]+sizes[i])
            nodes.append(range(tmp[i], tmp[i+1]))

        M = []
        labels = np.cumsum(sizes[:-1]).tolist()
        labels.insert(0, 0)
        G = nx.Graph()

        for i in range(num_clusters):
            M0 = nx.gnm_random_graph(sizes[i], n_edges_in_arr[i],
                                     seed=self.seed+i)
            tG = nx.Graph()
            for e in nx.edges(M0):
                tG.add_edge(*e, weight=wIn)
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
                    G.add_edge(*e, delay=wOut)
                    counter += 1

        adj = nx.to_numpy_array(G, dtype=int)
        adj_weighted = self.extract_attributes(G, 'weight')

        # density_clusters_info(G, clusters)
        print(nx.info(G))

        return adj, adj_weighted

    # ---------------------------------------------------------------#

    def generate_power_law_graph(self, N, exp, dmin, expect_n_edges,
                                 num_trial=100,
                                 tol=100):
        '''
        generate power law graph by given number of nodes, edges and 
        exponent
        '''

        def tune_dmin(x, N, exp, expect_n_edges):
            max_try = 50
            global G

            for i in range(max_try):
                sequence = generate_power_law_dist(N, exp, x, N-1)

                G = nx.configuration_model(sequence)
                G.remove_edges_from(G.selfloop_edges())
                G = nx.Graph(G)

                simple_seq = np.asarray([deg for (node, deg) in G.degree()])
                num_edges = int(np.sum(simple_seq)/2.0)

                if nx.is_connected(G):
                    # print "n = ", num_edges
                    break

            if i == (max_try-1):
                print("G is not connected")
                exit(0)

            return num_edges - expect_n_edges

        global G
        dmax = N-1

        for itr in range(num_trial):

            d_tuned = bisect(tune_dmin,
                             dmin-3, dmin+10,
                             args=(N, exp, expect_n_edges), xtol=0.01)
            simple_seq = np.asarray([deg for (node, deg) in G.degree()])
            num_edges = np.sum(simple_seq)/2.0
            error = np.abs(num_edges-expect_n_edges)
            print("dmin = %.2f, error = %d " % (d_tuned, error))

            if error < tol:
                return G, num_edges

        if itr == num_trial-1:
            print("could not find a proper graph with given properties!")
            exit(0)
    # ---------------------------------------------------------------#

    def from_matrix(self, fileName,
                    threshold=None,
                    plot_adj=False):
        """
        Returns adjacency matrix from text file.

        :param fileName: name of adjacency matrix file
        :param threshold: [float] if given, binarize the adjacency matrix
        :param plot_adj: [bool, optional(default=False)] if True plot the adjacency matrix
        :return: [ndarray] adjacency matrix
        """
        A = np.loadtxt(fileName, dtype=float)

        if binary:
            A = binarize(A, threshold)

        if plot_adj:
            self.plot_adjacency_matrix(A, **kwargs)

        return Adj
    # ---------------------------------------------------------------#

    def find_leader_node(self, adj_matrix, directed=True):

        if directed:
            Graph = nx.from_numpy_array(adj_matrix.T,
                                        create_using=nx.DiGraph())

            degree = list((dict(Graph.in_degree())).values())
        else:
            Graph = nx.from_numpy_array(adj_matrix.T)
            degree = list((dict(Graph.degree())).values())

        hub_index = degree.index(0)

        return hub_index
    # ---------------------------------------------------------------#

    def find_indices_with_distance(self, DiGraph, distance, source):
        """
        return index of nodes in given distance from source node.

        :param DiGraph: networkx directed graph.
        :param distance: [int] given distance from source node.
        :param source: index of given nodes as source, distance of other nodes calculated from this node.
        :return: index of nodes in given distance.
        """
        p = nx.shortest_path_length(DiGraph, source=source)
        distances = np.asarray(list(p.values()))

        return np.where(distances == distance)[0]
    # ---------------------------------------------------------------#

    def hierarchical_modular_network(self, n0,
                                     level,
                                     prob0,
                                     prob,
                                     alpha,
                                     weights,
                                     plot_adj=False):
        '''
        return a connected hierarchical modular network


        To construct a hierarchical and modular network (HMN) we started with a modular network of m modules, each having n0 nodes. In each module, nodes were connected with probability P1 (first level of hierarchy). Nodes in different modules were connected with the probability P2 (P2<P1) (second level of hierarchy. We add a copy of the previous network and connect the nodes belonging to these two different sets with a probability P3, where P3 < P2 (third level of hierarchy). Generally speaking, to construct a network with h levels of hierarchy, we repeated the above procedure h-1 times, with the hierarchical level-dependent probabilities, p_l=alpha q^(l-1) (l>1), where alpha is a constant, 0<q<1 and q<P1. The resulting network has 2^(h-1)m0 modules and 2^(h-1) m0 n0 nodes.

        :param n0: size of module at level 1
        :param level: number of levels
        :param prob0: probability of nodes in level 1.
        :param  prob: [float in [0,1]] this is q in : p_l=alpha q^(l-1)
        :param alpha: [float(default=1.0)] this is alpha in p_l=alpha q^(l-1)
        :param delays: [list of float or int] weights or any other attributes in each level
        :return: [2 ndarray] adjacency matrix and weighted matrix
        '''

        def probability(l, a=1, p=0.25):
            if l == 0:
                from sys import exit
                print("level shold be integer and > 0", l)
                exit(0)
            else:
                return a * p**l

        s = level
        n_modules = int(2**(s-1))  # number of modules
        N = int(n_modules*n0)  # number of nodes
        self.N = N

        # M0 = nx.complete_graph(n_M0)
        M0 = nx.erdos_renyi_graph(n0, prob0, seed=self.seed)
        for e in nx.edges(M0):
            # delays act in weight attribute
            M0.add_edge(*e, weight=weights[0])

        ps = [prob0]+[probability(i, alpha, prob) for i in range(1, s)]

        for l in range(1, s):
            if l == 1:
                M_pre = [M0] * n_modules
            else:
                M_pre = copy(M_next)
            M_next = []
            k = 0
            for ii in range(n_modules/(2**l)):
                step = 2**(l-1)
                tG = nx.convert_node_labels_to_integers(M_pre[k+1], step*n0)
                tG1 = nx.compose(M_pre[k], tG)
                edge = 0
                effort = 1
                # make sure that connected modules are not isolated
                while edge < 1:
                    # print "effort ", effort
                    effort += 1
                    for i in range(len(tG1)):
                        for j in range(i+1, len(tG1)):
                            if (i < step*n0) & (j > step*n0-1) & (np.random.rand() < ps[l]):
                                tG1.add_edge(i, j, weight=weights[l])
                                edge += 1

                M_next.append(tG1)
                k += 2
        self.G = M_next[0]

        if plot_adj:
            adj = nx.to_numpy_array(self.G, weight=None)
            self.plot_adjacency_matrix(adj, "con")

        adj_weighted = nx.to_numpy_array(self.G, weight='weight')

        adj = nx.to_numpy_array(self.G, weight=None)

        return adj, adj_weighted

    # ---------------------------------------------------------------#

    def multilevel(self, adj_matrix, directed=False, return_levels=False):
        """
        find communities of given weighted adjacency network

        :param adjMatrix: [2d array] adjacency matrix  
        :param directed: [bool(default=False)] choose directed or undirected network
        :param return_levels: if True, the communities at each level are returned in a list. If False, only the community structure with the best modularity is returned.
        :return: a list of VertexClustering objects, one corresponding to each level (if return_levels is True), or a VertexClustering corresponding to the best modularity.
        """
        conn_indices = np.where(adj_matrix)
        weights = adj_matrix[conn_indices]
        edges = zip(*conn_indices)
        G = igraph.Graph(edges=edges, directed=directed)
        G.es['weight'] = weights
        comm = G.community_multilevel(
            weights=weights, return_levels=return_levels)
        return comm
    # ---------------------------------------------------------------#

    def topological_sort(self, adj_matrix):
        """
        return topologicaly sorted network of given adjacency matrix.

        A topological sort is a nonunique permutation of the nodes such that an edge from u to v implies that u appears before v in the topological sort order.

        :param adj_atrix: [ndarray] given adjacency matrix of directed acyclic graph (DAG).
        :return: [ndarray] sorted adjacency matrix
        """

        At = copy(adj_matrix.T)
        G = nx.from_numpy_matrix(At, create_using=nx.DiGraph())
        assert (nx.is_directed_acyclic_graph(G)), "graph is not DAG"
        # new_node_label = nx.topological_sort(G, reverse=False)
        new_node_label = list(list(nx.topological_sort(G)))
        N = At.shape[0]
        An = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                An[i, j] = At[new_node_label[i], new_node_label[j]]
        G = nx.from_numpy_matrix(An, create_using=nx.DiGraph())
        assert (nx.is_directed_acyclic_graph(G)), "graph is not DAG"

        return An.T

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
#             self.plotAdjacencyMatrix(A, "A", title="Coupling weights")
#             self.plotAdjacencyMatrix(D, "D", title='Delays')

#         print(nx.info(G))

#         return A, D

# # ---------------------------------------------------------------------- #

    @staticmethod
    def rewiring_modular_graph(G,
                               clusters,
                               delay,
                               ens=1,
                               step_iteration=100,
                               threshold=0.185,
                               plot_adj=False,
                               path="./"):
        """
        rewire edges in a modular graph. 

        :param G: modular graph
        :param cluster: nested list of int inclusing the index of nodes in each cluster.
        :param ens: number of ensembling.
        :param step_iteration:  
        :param threshold: threshold of modularity, determine how many edges should be removed from clusters.
        :param path: path to save figure
        """

        # density_clusters_info(G, clusters)

        ofile = open("../data/text/lambda-"+str(ens)+".txt", "w")
        ffile = open("../data/text/fraction.txt", "a")

        if plot_adj:
            plotAdjacencyMatrix(nx.to_numpy_array(G),
                                "../data/fig/before-"+str(ens)+".png")

        print("lambda2/lambdan : %g" % lambdan_over_lambda2(G))

        n_nodes = nx.number_of_nodes(G)
        n_clusters = len(clusters)

        edges = nx.edges(G)
        for e in edges:
            G.add_edge(*e, delay=delay)

        n_edges = len(edges)
        edges = list(edges)

        a = [0]
        nodes = []
        for i in range(n_clusters):
            a.append(a[i]+clusters[i])
            nodes.append(range(a[i], a[i+1]))

        # put edges in clusters into a list
        edges_in = []
        for i in range(n_clusters):
            for e in edges:
                if ((e[0] in nodes[i]) & (e[1] in nodes[i])):
                    edges_in.append(e)
        shuffle(edges_in)
        n_edges_in = len(edges_in)

        counter = 0
        for it in range(n_edges_in):
            ei = np.random.randint(0, n_edges_in)
            edge = edges_in[ei]
            # print G.has_edge(*edge)
            G.remove_edge(*edge)
            del edges_in[ei]
            n_edges_in -= 1

            condition = False
            while True:
                e = np.random.randint(0, n_nodes, size=2)
                # print e
                if e[0] == e[1]:
                    continue
                else:
                    condition = check_edge_between_clusters(e, nodes)
                    if condition & (~G.has_edge(*e)):
                        G.add_edge(*e, delay=delay)
                        break
            if (it % 5) == 0:
                adj = nx.to_numpy_array(G)
                modularity = bct.modularity_und(adj)[1]
                frac = lambdan_over_lambda2(G)

                ofile.write("%6d %12.6f %12.6f\n" %
                            (it, frac, modularity))
                if ((it % step_iteration) == 0):
                    Adj = nx.to_numpy_array(G)
                    D = extract_attributes(G, "delay")
                    np.savetxt(str("../data/text/C-%d-%d.txt" % (counter, ens)),
                               Adj, fmt='%d')
                    np.savetxt(str("../data/text/D-%d-%d.txt" % (counter, ens)),
                               D, fmt='%d')
                    counter += 1
                    ffile.write("%10.3f" % frac)

                if modularity < threshold:
                    break
        ffile.write("\n")
        ffile.close()

        ofile.close()
        if plot_adj:
            plotAdjacencyMatrix(nx.to_numpy_array(
                G), path+"after-"+str(ens)+".png")

        # density_clusters_info(G, clusters)
# ---------------------------------------------------------------------- #

    @staticmethod
    def check_edge_between_clusters(e, nodes):
        """check if given edge is between clusters.

        :param e: [int, int] given edge 
        :param nodes: nested list (list of list of int) including the index of nodes in each cluster.
        :return: [bool] True if edge be between the clusters
        """
        # print e[0], e[1]
        n_clusters = len(nodes)
        s = range(n_clusters)
        shuffle(s)
        for i in s:
            if ((e[0] in nodes[i]) & (e[1] not in nodes[i])) |\
                    ((e[0] not in nodes[i]) & (e[1] in nodes[i])):
                return True

        return False

# ---------------------------------------------------------------------- #

    @staticmethod
    def clusters_info_modular(G, clusters, print_result=True):
        """
        Returns the properties of clusters in a modular network.

        :param G: undirected Graph of modular network
        :param clusters: list of list of int inclusing the index of nodes in each cluster.
        :param print_results: [bool] if True print the results on the screen.
        :return: [int, int] total number of edges and number of edges between clusters.
        """

        N = nx.number_of_nodes(G)
        n_clusters = len(clusters)

        edges = G.edges()
        num_edges = nx.number_of_edges(G)
        num_edges_in = 0

        a = [0]
        nodes = []
        for i in range(n_clusters):
            a.append(a[i]+clusters[i])
            nodes.append(range(a[i], a[i+1]))

        if print_result:
            print("="*70)
            print('%s%13s%15s%15s%15s' % (
                'index', 'size', 'density',
                'n_edges_in', 'n_edges_out'))

        num_edges_in_total = 0
        num_edges_out_total = 0
        for i in range(n_clusters):
            num_e_in = 0
            num_e_out = 0
            for e in edges:
                if ((e[0] in nodes[i]) & (e[1] in nodes[i])):
                    num_e_in += 1
                elif ((e[0] in nodes[i]) & (e[1] not in nodes[i])) |\
                        ((e[0] not in nodes[i]) & (e[1] in nodes[i])):
                    num_e_out += 1

            num_edges_in_total += num_e_in
            num_edges_out_total += num_e_out
            density_in = 2.0 * num_e_in / float(clusters[i]*(clusters[i]-1))

            if print_result:
                print('%1d%15d%15.2f%15d%15d' % (i, clusters[i],
                                                 density_in, num_e_in, num_e_out))
        edges_between = num_edges-num_edges_in_total
        if print_result:
            print("number of edges : %d" % len(edges))
            print("number of edges in clusters : %d" % num_edges_in_total)
            print("number of edges between clusters : %d" %
                  (num_edges_out_total/2))
            print("fraction of in links : %g " % (
                num_edges_in_total / float(num_edges)))
            print("fraction of between links : %g " % (
                edges_between/float(num_edges)))

        return (num_edges_in_total, edges_between)
# --------------------------------------------------------------#


# def lambdan_over_lambda2(G):

#     L = nx.laplacian_matrix(G)
#     eig = np.linalg.eigvals(L.A)
#     eig = np.sort(eig)
#     r = eig[-1]/float(eig[1])

#     return r

# # --------------------------------------------------------------#
