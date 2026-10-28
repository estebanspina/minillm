"""Pruebas mínimas con unittest (sin pytest)."""
import unittest
import numpy as np
from minillm import softmax, preparar_datos, DATA

class MiniLLMTests(unittest.TestCase):
    def test_softmax_probabilidades(self):
        salida = softmax(np.array([[2.0, 1.0, 0.0], [0.0, 0.0, 0.0]]))
        self.assertTrue(np.allclose(salida.sum(axis=1), 1))
        self.assertTrue(np.all(salida >= 0))
    def test_softmax_estable(self):
        salida = softmax(np.array([[1000.0, 999.0]]))
        self.assertTrue(np.isfinite(salida).all())
    def test_corpus(self):
        texto, vocab, ids = preparar_datos(DATA)
        self.assertGreater(len(texto), 2)
        self.assertEqual(len(ids), len(texto))
        self.assertEqual(len(vocab), len(set(texto)))

if __name__ == "__main__":
    unittest.main()
