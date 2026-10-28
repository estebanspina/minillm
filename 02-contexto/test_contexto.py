import unittest
import numpy as np
from minillm_contexto import datos, iniciar, forward, perdida, paso, softmax

class TestContexto(unittest.TestCase):
    def test_ventanas(self):
        vocab, X, y = datos("ababa", 2)
        self.assertEqual(X.shape, (3,2))
        self.assertEqual("".join(vocab[i] for i in X[0]), "ab")
        self.assertEqual(vocab[y[0]], "a")
    def test_softmax(self):
        p = softmax(np.array([[1000., 999.]]))
        self.assertTrue(np.isfinite(p).all())
        self.assertAlmostEqual(p.sum(), 1.)
    def test_entrenamiento(self):
        rng = np.random.default_rng(5)
        p = iniciar(3, 2, 4, 8, rng)
        X = np.array([[0,1],[1,2],[2,0]]); y = np.array([2,0,1])
        antes = perdida(forward(X,p)[0], y)
        for _ in range(100):
            paso(X,y,p,.3)
        despues = perdida(forward(X,p)[0], y)
        self.assertLess(despues, antes)

if __name__ == "__main__":
    unittest.main()
