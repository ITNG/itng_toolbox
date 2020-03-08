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
from scipy.optimize import bisect, newton, brenth
from itng.drawing import Drawing
import itng.graphUtility

# pl.switch_backend('agg')


class Generators:
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

    def complete_graph(self, N,
                       create_using=nx.Graph()
                       ):
        ''' 
        Return the complete graph with N nodes.

        :param N: number of nodes        
        :param kwargs: argument pass to ploting the adjacency matrix
        :return: networkx graph
        '''

        G = nx.complete_graph(N, create_using=create_using)

        return G
    # ---------------------------------------------------------------#

    def gnp_random_graph(self, n, p, directed=False):
        ''' 
        Returns Erdos Renyi graph, given size and probability 

        :param n: [int] number of nodes
        :param p: [float] probability of existing an edge
        :param directed: bool, optional (default=False) If True, this function returns a directed graph.
        :return: networkx graph

        See also of :py:meth:`gnm_random_grph() <Generators.gnm_random_graph>`.
        '''

        G = nx.gnp_random_graph(n, p, seed=self.seed, directed=directed)

        return G
    # ---------------------------------------------------------------#

    def gnm_random_graph(self, n, m, directed=False):
        ''' 
        Returns Erdos Renyi network, given size and number of edges

        :param n: [int] number of nodes
        :param m: [int] number of edges
        :param directed: bool, optional (default=False) If True, this function returns a directed graph.
        :return: networkx graph

        See also of :py:meth:`gnp_random_grph() <Generators.gnp_random_graph>`.
        '''

        G = nx.gnm_random_graph(n, m, seed=self.seed, directed=False)

        return G
    # ---------------------------------------------------------------#

    def modular_graph(self, N,
                      pIn,
                      pOut,
                      sizes,
                      wIn,
                      wOut,
                      **kwargs):
        ''' 
        returns a modular graphs

        :param N: [int] number of nodes
        :param pIn: [int, float or list of numbers] probability of existing an edge inside clusters
        :param pOut: [int, float] probability of existing an edge between clusters
        :param sizes: [list of int] size of clusters in network
        :param wIn: weight of edges inside clusters
        :param wOut: weight of edges between clusters
        :return: weighted networkx graph
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
                    # A[i, j] = A[j, i] = 1.0
                    W[i, j] = W[j, i] = wOut

        # empty inside the clusters
        s = 0
        for k in range(num_cluster):
            if k > 0:
                s += sizes[k-1]
            for i in range(s, (sizes[k]+s)):
                for j in range(i+1, (sizes[k]+s)):
                    # A[i, j] = A[j, i] = 0.0
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
                        # A[i, j] = A[j, i] = 1.0
                        W[i, j] = W[j, i] = wIn

        G = nx.from_numpy_array(W)

        return G
    # ---------------------------------------------------------------#

    def modular_network_fixed_num_edges(self, N,
                                        n_edges_in,
                                        n_edges_between,
                                        sizes,
                                        wIn,
                                        wOut,
                                        verbosity=True,
                                        ):
        """
        Returns a modular graph with given number of edges

        :param N: [int] number of nodes
        :param n_edges_in: [int] number of edges inside the clusters
        :param n_edges_between: [int] number of edges between clusters
        :param sizes: [list of int] size of each module 
        :param wIn: [float] weight of edges inside the clusters
        :param wOut: [float] weight of edges between the clusters
        :return: networkx graph
        """

        if self.seed is None:
            seed = np.random.randint(1e6)
        else:
            seed = self.seed

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
                                     seed=seed+i)
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
                condition = graphUtility.check_edge_between_clusters(e, nodes)
                if condition & (~G.has_edge(*e)):
                    G.add_edge(*e, delay=wOut)
                    counter += 1

        # adj = nx.to_numpy_array(G, dtype=int)
        # adj_weighted = self.extract_attributes(G, 'weight')

        # density_clusters_info(G, clusters)
        if verbosity:
            print(nx.info(G))

        return G

    # ---------------------------------------------------------------#

    def power_law_graph(self, N, exp, dmin, expect_n_edges,
                        num_trial=100, tol=100):
        '''
        generate power law [connected] graph by given number of nodes, edges and 
        exponent.

        :param N:
        :param exp: [float] negative exponent in power low graph, typical value is -2.0 to -3.0
        :param dmin: minimum degree of a node.
        :param expect_n_edges: expected number of edges in graph
        :param num_trial: number of trial to try find a connected graph
        :param tol: the tolerance number of edges versus the expected given number of edges.
        :return: networkx graph
        '''

        def tune_dmin(x, G, N, exp, expect_n_edges, max_num_try=50):
            max_try = 50

            for i in range(max_num_try):
                sequence = generate_power_law_dist(N, exp, x, N-1)

                G = nx.configuration_model(sequence)
                G.remove_edges_from(G.selfloop_edges())
                G = nx.Graph(G)

                simple_seq = np.asarray([deg for (node, deg) in G.degree()])
                num_edges = int(np.sum(simple_seq)/2.0)

                if nx.is_connected(G):
                    # print "n = ", num_edges
                    break

            if i == (max_num_try-1):
                print("G is not connected")
                exit(0)

            return num_edges - expect_n_edges

        G = nx.Graph()
        dmax = N-1

        for itr in range(num_trial):

            # tune minimum degree of the node
            d_tuned = bisect(tune_dmin,
                             dmin-3, dmin+10, # sweep the interval around dmin
                             args=(G, N, exp, expect_n_edges),
                             xtol=0.01)

            simple_seq = np.asarray([deg for (node, deg) in G.degree()])
            num_edges = np.sum(simple_seq)/2.0
            error = np.abs(num_edges - expect_n_edges)
            
            if verbocity:
                print ("dmin = %.2f, error = %d, n_edges = %d " % (
                    d_tuned, error, num_edges))

            # check if number of edges is close to the expected number of edges
            if error < tol:
                return G

        if itr == num_trial-1:
            print("could not find a proper graph with given properties!")
            exit(0)
    
    # ---------------------------------------------------------------#

    def hierarchical_modular_graph(self, n0,
                                     level,
                                     prob0,
                                     prob,
                                     alpha,
                                     weights):
        '''
        return a connected hierarchical modular network


        To construct a hierarchical and modular network (HMN) we started with a modular network of m modules, each having n0 nodes. In each module, nodes were connected with probability P1 (first level of hierarchy). Nodes in different modules were connected with the probability P2 (P2<P1) (second level of hierarchy. We add a copy of the previous network and connect the nodes belonging to these two different sets with a probability P3, where P3 < P2 (third level of hierarchy). Generally speaking, to construct a network with h levels of hierarchy, we repeated the above procedure h-1 times, with the hierarchical level-dependent probabilities, p_l=alpha q^(l-1) (l>1), where alpha is a constant, 0<q<1 and q<P1. The resulting network has 2^(h-1)m0 modules and 2^(h-1) m0 n0 nodes.

        :param n0: size of module at level 1
        :param level: number of levels
        :param prob0: probability of nodes in level 1.
        :param  prob: [float in [0,1]] this is q in : p_l=alpha q^(l-1)
        :param alpha: [float(default=1.0)] this is alpha in p_l=alpha q^(l-1)
        :param delays: [list of float or int] weights or any other attributes in each level
        :return: weighted networkx graph
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
            for ii in range(n_modules//(2**l)):
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
        G = M_next[0]

        # adj_weighted = nx.to_numpy_array(self.G, weight='weight')
        # adj = nx.to_numpy_array(self.G, weight=None)

        return G

    # ---------------------------------------------------------------#
