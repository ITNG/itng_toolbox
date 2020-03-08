import numpy as np
import networkx as nx
from random import shuffle


def extract_attributes(G, attr=None):
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


def print_adj_matrix(G, file_name=None, fmt="%d", binary_file=False):
    """
    print the adjacency matrix of given graph

    :param G: networkx graph
    :param file_name: optional(default=None), if given save to file
    :param fmt: optional, format of numbeers in weighted adjacency matrix, "%d" for integer and e.g. "%10.3f", "%g", ... for float numbers. 
    :param binary_file: [bool] if True, save npz file in binary format.
    """

    adj = nx.to_numpy_array(G)

    if file_name:  # print to file

        if binary_file:
            np.savez(file_name, a=adj)

        else:
            np.savetxt(file_name, adj, fmt=fmt)

    else:  # print on the screen
        for row in M:
            for val in row:
                print('%.0f' % val),
            print ()

# ---------------------------------------------------------------#


def binarize(adj, threshold):
    """
    binarize the given 2d numpy array

    :param data: [2d numpy array] given array.
    :param threshold: [float] threshold value.
    :return: [2d int numpy array] binarized array.
    """

    adj = np.asarray(adj)
    upper, lower = 1, 0
    adj = np.where(adj >= threshold, upper, lower)
    return adj
# --------------------------------------------------------------#


def lambdan_over_lambda2(G):
    """
    calculate the fraction of lambda_n over lambda_2.
    lambda_i s are the eigen values of laplacian matrix

    """

    L = nx.laplacian_matrix(G)
    eig = np.linalg.eigvals(L.A)
    eig = np.sort(eig)
    r = eig[-1]/float(eig[1])

    return r

# --------------------------------------------------------------#


def check_edge_between_clusters(e, nodes):
    """check if given edge is between clusters.

    :param e: [int, int] given edge 
    :param nodes: nested list (list of list of int) including the index of nodes in each cluster.
    :return: [bool] True if edge be between the clusters
    """
    # print e[0], e[1]
    n_clusters = len(nodes)
    s = list(range(n_clusters))
    shuffle(s)
    for i in s:
        if ((e[0] in nodes[i]) & (e[1] not in nodes[i])) |\
                ((e[0] not in nodes[i]) & (e[1] in nodes[i])):
            return True

    return False
# --------------------------------------------------------------#


def topological_sort(self, adj_matrix):
    """
    return topologicaly sorted network of given directed acyclic graph.

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
# ------------------------------------------------------------------#


def find_leader_node(self, adj, directed=True):
    """
    find the leader node in adj matrix of a directed acyclic graph
    leader node is the node with in degree of zero.

    :param adj:
    :directed: [bool] if True consider a directed graph
    :return: index of leader node
    """

    if directed:
        Graph = nx.from_numpy_array(adj.T,
                                    create_using=nx.DiGraph())

        degree = list((dict(Graph.in_degree())).values())
    else:
        Graph = nx.from_numpy_array(adj.T)
        degree = list((dict(Graph.degree())).values())

    leader_index = degree.index(0)

    return leader_index
# ---------------------------------------------------------------#


def multilevel(adj_matrix, directed=False, return_levels=False):
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


def find_indices_with_distance(DiGraph, distance, source):
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

# ------------------------------------------------------------------#


def clusters_info_modular(G, clusters, verbosity=True):
    """
    Returns the properties of clusters in a modular network.

    :param G: undirected Graph of modular network
    :param clusters: list of list of int inclusing the index of nodes in each cluster.
    :param verbosity: [bool] if True print the results on the screen.
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

    if verbosity:
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

        if verbosity:
            print('%1d%15d%15.2f%15d%15d' % (i, clusters[i],
                                             density_in, num_e_in, num_e_out))
    edges_between = num_edges-num_edges_in_total
    if verbosity:
        print("number of edges : %d" % len(edges))
        print("number of edges in clusters : %d" % num_edges_in_total)
        print("number of edges between clusters : %d" %
              (num_edges_out_total/2))
        print("fraction of in links : %g " % (
            num_edges_in_total / float(num_edges)))
        print("fraction of between links : %g " % (
            edges_between/float(num_edges)))

    return (num_edges_in_total, edges_between)
# ------------------------------------------------------------------#


def calculate_NMI(self, comm1, comm2, method="nmi"):
    """
    Compares two community structures

    :param comm1: the first community structure as a membership list or as a Clustering object.
    :param comm2: the second community structure as a membership list or as a Clustering object.
    :param method: [string] defaults to ["nmi"] the measure to use. "vi" or "meila" means the variation of information metric of Meila (2003), "nmi" or "danon" means the normalized mutual information as defined by Danon et al (2005), "split-join" means the split-join distance of van Dongen (2000), "rand" means the Rand index of Rand (1971), "adjusted_rand" means the adjusted Rand index of Hubert and Arabie (1985).
    :return: [float] the calculated measure.

    Reference: 

    -  Meila M: Comparing clusterings by the variation of information. In: Scholkopf B, Warmuth MK (eds). Learning Theory and Kernel Machines: 16th Annual Conference on Computational Learning Theory and 7th Kernel Workship, COLT/Kernel 2003, Washington, DC, USA. Lecture Notes in Computer Science, vol. 2777, Springer, 2003. ISBN: 978-3-540-40720-1.

    -  Danon L, Diaz-Guilera A, Duch J, Arenas A: Comparing community structure identification. J Stat Mech P09008, 2005.

    -  van Dongen D: Performance criteria for graph clustering and Markov cluster experiments. Technical Report INS-R0012, National Research Institute for Mathematics and Computer Science in the Netherlands, Amsterdam, May 2000.

    -  Rand WM: Objective criteria for the evaluation of clustering methods. J Am Stat Assoc 66(336):846-850, 1971.

    -  Hubert L and Arabie P: Comparing partitions. Journal of Classification 2:193-218, 1985.
    """

    nmi = igraph.compare_communities(
        communities1, comm2, method='nmi', remove_none=False)
    return nmi
# ------------------------------------------------------------------#
