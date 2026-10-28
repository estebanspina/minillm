import unittest
from bigramas import aprender, distribucion, generar

class TestBigramas(unittest.TestCase):
    def test_conteos(self):
        self.assertEqual(aprender("ababa")["a"]["b"], 2)
        self.assertEqual(aprender("ababa")["b"]["a"], 2)

    def test_probabilidades(self):
        dist = dict(distribucion(aprender("abac"), "a"))
        self.assertAlmostEqual(dist["b"], 0.5)
        self.assertAlmostEqual(dist["c"], 0.5)

    def test_reproducibilidad(self):
        modelo = aprender("abracadabra")
        self.assertEqual(generar(modelo, "a", semilla=9), generar(modelo, "a", semilla=9))

if __name__ == "__main__":
    unittest.main()
