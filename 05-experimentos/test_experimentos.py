import unittest
from experimentos import extraer_metricas

class TestExperimentos(unittest.TestCase):
    def test_parsear_metricas(self):
        salida = "paso    1 | train=3.500 | valid=3.100\npaso   10 | train=2.000 | valid=2.400"
        self.assertEqual(extraer_metricas(salida), [(1,3.5,3.1),(10,2.,2.4)])
    def test_salida_vacia(self):
        self.assertEqual(extraer_metricas("no hay métricas"), [])

if __name__ == "__main__":
    unittest.main()
