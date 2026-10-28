import unittest
import torch
from mini_transformer import MiniTransformer

class TestMiniTransformer(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.modelo = MiniTransformer(12, contexto=8, dimension=16, cabezas=2).eval()

    def test_forma(self):
        resultado = self.modelo(torch.randint(0, 12, (2, 5)))
        self.assertEqual(tuple(resultado.shape), (2, 5, 12))

    def test_no_mira_el_futuro(self):
        a = torch.tensor([[1,2,3,4]])
        b = torch.tensor([[1,2,9,8]])
        with torch.no_grad():
            x = self.modelo(a)
            y = self.modelo(b)
        torch.testing.assert_close(x[:, :2], y[:, :2], atol=1e-6, rtol=1e-6)

    def test_gradientes(self):
        self.modelo.train()
        pred = self.modelo(torch.randint(0, 12, (2, 5)))
        pred.sum().backward()
        self.assertIsNotNone(self.modelo.qkv.weight.grad)
        self.assertTrue(torch.isfinite(self.modelo.qkv.weight.grad).all())

if __name__ == "__main__":
    unittest.main()
