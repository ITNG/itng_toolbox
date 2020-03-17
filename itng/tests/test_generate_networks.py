import numpy as np
import unittest
import sys
sys.path.insert(0, "../")
import networks
import networkx as nx



class TestModules(unittest.TestCase):

    def test_complete_graph(self):

        g = networks.Generators()
        G = g.complete_graph(2)
        adj = nx.to_numpy_array(G)
        expected = np.array([[0,1],[1,0]], dtype=float)
        self.assertEqual(np.array_equal(adj, expected), True)


if __name__ == "__main__":
    unittest.main()
