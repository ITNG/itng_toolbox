import numpy as np
import unittest
import sys
sys.path.insert(0, "../")
import generateNetworks



class TestModules(unittest.TestCase):

    def test_complete_graph(self):

        G = generateNetworks.graphGenerator()
        M = G.complete_graph(2)
        expected = np.array([[0,1],[1,0]], dtype=float)
        self.assertEqual(np.array_equal(M, expected), True)


if __name__ == "__main__":
    unittest.main()
