"""Modelo neuronal de caracteres con una ventana de contexto (NumPy + CPU)."""
import argparse
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent
CORPUS = BASE.parent / "01-red-neuronal" / "datos" / "cuentos.txt"
PESOS = BASE / "modelo_contexto.npz"

def datos(texto, contexto):
    if contexto < 1 or len(texto) <= contexto:
        raise ValueError("Se requieren más caracteres que la ventana de contexto")
    vocab = sorted(set(texto))
    mapa = {ch: i for i, ch in enumerate(vocab)}
    ids = np.array([mapa[ch] for ch in texto], dtype=np.int64)
    X = np.stack([ids[i:i+contexto] for i in range(len(ids)-contexto)])
    y = ids[contexto:]
    return vocab, X, y

def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=-1, keepdims=True)

def iniciar(vocab_size, contexto, dim, ocultas, rng):
    return {
        "E": rng.normal(0, .1, (vocab_size, dim)),
        "W1": rng.normal(0, .1, (contexto * dim, ocultas)),
        "b1": np.zeros(ocultas),
        "W2": rng.normal(0, .1, (ocultas, vocab_size)),
        "b2": np.zeros(vocab_size),
    }

def forward(X, p):
    plano = p["E"][X].reshape(len(X), -1)
    h = np.tanh(plano @ p["W1"] + p["b1"])
    prob = softmax(h @ p["W2"] + p["b2"])
    return prob, (plano, h)

def perdida(prob, y):
    return float(-np.log(prob[np.arange(len(y)), y] + 1e-12).mean())

def paso(X, y, p, tasa):
    prob, (plano, h) = forward(X, p)
    loss = perdida(prob, y)
    dz2 = prob.copy()
    dz2[np.arange(len(y)), y] -= 1
    dz2 /= len(y)
    dW2 = h.T @ dz2
    db2 = dz2.sum(axis=0)
    dz1 = (dz2 @ p["W2"].T) * (1 - h*h)
    dW1 = plano.T @ dz1
    db1 = dz1.sum(axis=0)
    dplano = dz1 @ p["W1"].T
    dE = np.zeros_like(p["E"])
    np.add.at(dE, X, dplano.reshape(len(X), X.shape[1], -1))
    for clave, grad in (("E", dE), ("W1", dW1), ("b1", db1), ("W2", dW2), ("b2", db2)):
        p[clave] -= tasa * grad
    return loss

def entrenar(args):
    texto = CORPUS.read_text(encoding="utf-8").lower()
    # Separación temporal: el conjunto de validación no comparte objetivos con entrenamiento.
    corte = int(len(texto) * .8)
    if corte <= args.contexto or len(texto)-corte <= args.contexto:
        raise SystemExit("Corpus demasiado pequeño para la ventana")
    vocab = sorted(set(texto))
    mapa = {c: i for i, c in enumerate(vocab)}
    def ventanas(parte):
        ids = np.array([mapa[c] for c in parte], dtype=np.int64)
        return np.stack([ids[i:i+args.contexto] for i in range(len(ids)-args.contexto)]), ids[args.contexto:]
    X, y = ventanas(texto[:corte])
    Xv, yv = ventanas(texto[corte:])
    rng = np.random.default_rng(args.semilla)
    p = iniciar(len(vocab), args.contexto, args.dimension, args.ocultas, rng)
    print(f"Contexto={args.contexto}, parámetros={sum(a.size for a in p.values())}, ventanas={len(X)}")
    for i in range(1, args.pasos + 1):
        idx = rng.integers(0, len(X), size=args.lote)
        train_loss = paso(X[idx], y[idx], p, args.tasa)
        if i == 1 or i % max(1, args.pasos//10) == 0:
            val_loss = perdida(forward(Xv, p)[0], yv)
            print(f"paso {i:4} | entrenamiento={train_loss:.4f} | validación={val_loss:.4f}")
    np.savez(PESOS, **p, vocabulario=np.array(vocab), contexto=np.array(args.contexto))
    print(f"Pesos guardados: {PESOS}")

def generar(args):
    if not PESOS.exists():
        raise SystemExit("Primero entrená el modelo")
    with np.load(PESOS, allow_pickle=False) as z:
        p = {k: z[k] for k in ("E", "W1", "b1", "W2", "b2")}
        vocab = list(z["vocabulario"])
        contexto = int(z["contexto"])
    mapa = {c:i for i,c in enumerate(vocab)}
    inicio = args.inicio.lower()
    if not inicio or any(c not in mapa for c in inicio):
        raise SystemExit("El inicio debe contener solo caracteres del corpus")
    relleno = " " if " " in mapa else vocab[0]
    secuencia = list((relleno * contexto + inicio)[-contexto:])
    salida = list(inicio)
    rng = np.random.default_rng(args.semilla)
    for _ in range(args.longitud):
        X = np.array([[mapa[c] for c in secuencia]])
        prob = softmax(np.log(forward(X, p)[0][0] + 1e-12) / args.temperatura)
        siguiente = vocab[int(rng.choice(len(vocab), p=prob))]
        salida.append(siguiente)
        secuencia = (secuencia + [siguiente])[-contexto:]
    print("".join(salida))

def main():
    ap = argparse.ArgumentParser(description="MiniLLM: contexto de varios caracteres")
    sub = ap.add_subparsers(dest="comando", required=True)
    tr = sub.add_parser("entrenar")
    tr.add_argument("--pasos", type=int, default=1500)
    tr.add_argument("--contexto", type=int, default=4)
    tr.add_argument("--dimension", type=int, default=8)
    tr.add_argument("--ocultas", type=int, default=32)
    tr.add_argument("--lote", type=int, default=64)
    tr.add_argument("--tasa", type=float, default=.3)
    tr.add_argument("--semilla", type=int, default=42)
    ge = sub.add_parser("generar")
    ge.add_argument("--inicio", default="el ")
    ge.add_argument("--longitud", type=int, default=200)
    ge.add_argument("--temperatura", type=float, default=.8)
    ge.add_argument("--semilla", type=int, default=7)
    args = ap.parse_args()
    if args.comando == "entrenar":
        if min(args.pasos,args.contexto,args.dimension,args.ocultas,args.lote,args.tasa) <= 0:
            ap.error("Los parámetros deben ser positivos")
        entrenar(args)
    else:
        if args.longitud < 0 or args.temperatura <= 0:
            ap.error("Longitud no negativa y temperatura positiva")
        generar(args)

if __name__ == "__main__":
    main()
