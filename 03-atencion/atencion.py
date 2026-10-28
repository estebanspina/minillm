"""Self-attention causal educativa en NumPy, sin entrenamiento ni GPU."""
import argparse
import numpy as np

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def atencion_causal(X, Wq, Wk, Wv):
    """X: (tokens, dimensiones). Retorna salida, pesos de atención."""
    if X.ndim != 2:
        raise ValueError("X debe ser una matriz (tokens, dimensiones)")
    n, d = X.shape
    if n == 0 or d == 0:
        raise ValueError("X no puede estar vacío")
    if any(W.shape != (d, d) for W in (Wq, Wk, Wv)):
        raise ValueError("Las matrices Wq, Wk y Wv deben ser (d, d)")
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    scores = Q @ K.T / np.sqrt(d)
    mascara_futuro = np.triu(np.ones((n, n), dtype=bool), k=1)
    scores = np.where(mascara_futuro, -np.inf, scores)
    pesos = softmax(scores)
    return pesos @ V, pesos

def ejemplo(texto, dimension=8, semilla=7):
    if not texto:
        raise ValueError("El texto no puede estar vacío")
    vocab = sorted(set(texto))
    ids = {c:i for i,c in enumerate(vocab)}
    rng = np.random.default_rng(semilla)
    embeddings = rng.normal(0, .2, (len(vocab), dimension))
    X = embeddings[[ids[c] for c in texto]]
    Wq = rng.normal(0, .2, (dimension, dimension))
    Wk = rng.normal(0, .2, (dimension, dimension))
    Wv = rng.normal(0, .2, (dimension, dimension))
    salida, pesos = atencion_causal(X, Wq, Wk, Wv)
    return salida, pesos

def main():
    parser = argparse.ArgumentParser(description="Demostración de atención causal")
    parser.add_argument("--texto", default="hola")
    parser.add_argument("--dimension", type=int, default=8)
    args = parser.parse_args()
    if args.dimension < 1 or not args.texto:
        parser.error("Se necesita texto no vacío y dimensión positiva")
    salida, pesos = ejemplo(args.texto, args.dimension)
    print("Cada fila es un token que consulta; cada columna, un token atendido.")
    print("Los valores encima de la diagonal son cero: no se ve el futuro.")
    print("    " + " ".join(f"{i:>6}" for i in range(len(args.texto))))
    for i, fila in enumerate(pesos):
        print(f"{i:>3} " + " ".join(f"{v:6.3f}" for v in fila))
    print(f"Forma de la salida: {salida.shape}")

if __name__ == "__main__":
    main()
