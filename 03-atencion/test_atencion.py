import unittest
import numpy as np
from atencion import atencion_causal, ejemplo

class TestAtencion(unittest.TestCase):
    def test_mascara_y_filas(self):
        _, p = ejemplo("hola")
        self.assertTrue(np.all(p[np.triu_indices(4, k=1)] == 0))
        self.assertTrue(np.allclose(p.sum(axis=1), 1))
    def test_no_ver_futuro(self):
        ident = np.eye(3)
        a = np.array([[1.,0,0],[0,1,0],[0,0,1.]])
        b = a.copy()
        b[2] = [99,99,99]
        sa,_ = atencion_causal(a, ident,ident,ident)
        sb,_ = atencion_causal(b, ident,ident,ident)
        np.testing.assert_allclose(sa[:2], sb[:2])
    def test_formas(self):
        out, weights = ejemplo("casa", dimension=6)
        self.assertEqual(out.shape, (4,6))
        self.assertEqual(weights.shape, (4,4))

if __name__ == "__main__":
    unittest.main()
