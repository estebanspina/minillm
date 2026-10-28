"""MiniLLM: modelo neuronal de caracteres, sin pesos preentrenados."""
import argparse
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent
DATA = BASE / "datos" / "cuentos.txt"
MODEL = BASE / "modelo.npz"

def preparar_datos(ruta):
    texto = ruta.read_text(encoding="utf-8").lower()
    vocabulario = sorted(set(texto))
    mapa = {c: i for i, c in enumerate(vocabulario)}
    ids = np.array([mapa[c] for c in texto], dtype=np.int64)
    return texto, vocabulario, ids

def softmax(logits):
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)

def entrenar(pasos, tasa, dimension, lote, semilla):
    texto, vocabulario, ids = preparar_datos(DATA)
    if len(ids) < 2:
        raise ValueError("El corpus necesita al menos 2 caracteres")
    rng = np.random.default_rng(semilla)
    v = len(vocabulario)
    E = rng.normal(0, 0.1, size=(v, dimension))
    W = rng.normal(0, 0.1, size=(dimension, v))
    b = np.zeros(v)
    print(f"Caracteres: {len(texto)} | vocabulario: {v} | parámetros: {E.size + W.size + b.size}")

    for paso in range(1, pasos + 1):
        posiciones = rng.integers(0, len(ids) - 1, size=lote)
        x, y = ids[posiciones], ids[posiciones + 1]
        h = E[x]
        logits = h @ W + b
        probs = softmax(logits)
        perdida = -np.log(probs[np.arange(lote), y] + 1e-12).mean()

        # Gradientes de entropía cruzada y softmax, calculados a mano.
        dlogits = probs.copy()
        dlogits[np.arange(lote), y] -= 1
        dlogits /= lote
        dW = h.T @ dlogits
        db = dlogits.sum(axis=0)
        dh = dlogits @ W.T
        dE = np.zeros_like(E)
        np.add.at(dE, x, dh)

        E -= tasa * dE
        W -= tasa * dW
        b -= tasa * db
        if paso == 1 or paso % max(1, pasos // 10) == 0:
            print(f"paso {paso}/{pasos} | pérdida = {perdida:.4f}")

    np.savez(MODEL, E=E, W=W, b=b, vocabulario=np.array(vocabulario))
    print(f"Modelo guardado: {MODEL}")

def generar(inicio, longitud, temperatura, semilla):
    if not MODEL.exists():
        raise SystemExit("Primero ejecutá: python minillm.py entrenar")
    if temperatura <= 0:
        raise SystemExit("La temperatura debe ser positiva")
    with np.load(MODEL, allow_pickle=False) as datos:
        E, W, b = datos["E"], datos["W"], datos["b"]
        vocabulario = list(datos["vocabulario"])
    mapa = {c: i for i, c in enumerate(vocabulario)}
    inicio = inicio.lower()
    desconocidos = sorted(set(inicio) - set(mapa))
    if desconocidos:
        raise SystemExit(f"Caracteres desconocidos: {desconocidos}")
    if not inicio:
        inicio = " " if " " in mapa else vocabulario[0]
    rng = np.random.default_rng(semilla)
    salida = list(inicio)
    actual = mapa[inicio[-1]]
    for _ in range(longitud):
        probs = softmax((E[actual] @ W + b) / temperatura)
        actual = int(rng.choice(len(vocabulario), p=probs))
        salida.append(vocabulario[actual])
    print("".join(salida))

def main():
    parser = argparse.ArgumentParser(description="MiniLLM: red neuronal de caracteres")
    sub = parser.add_subparsers(dest="comando", required=True)
    train = sub.add_parser("entrenar")
    train.add_argument("--pasos", type=int, default=1500)
    train.add_argument("--tasa", type=float, default=0.4)
    train.add_argument("--dimension", type=int, default=16)
    train.add_argument("--lote", type=int, default=64)
    train.add_argument("--semilla", type=int, default=42)
    gen = sub.add_parser("generar")
    gen.add_argument("--inicio", default="el ")
    gen.add_argument("--longitud", type=int, default=200)
    gen.add_argument("--temperatura", type=float, default=0.8)
    gen.add_argument("--semilla", type=int, default=7)
    args = parser.parse_args()
    if args.comando == "entrenar":
        if min(args.pasos, args.tasa, args.dimension, args.lote) <= 0:
            parser.error("Los parámetros de entrenamiento deben ser positivos")
        entrenar(args.pasos, args.tasa, args.dimension, args.lote, args.semilla)
    else:
        if args.longitud < 0:
            parser.error("La longitud debe ser no negativa")
        generar(args.inicio, args.longitud, args.temperatura, args.semilla)

if __name__ == "__main__":
    main()
