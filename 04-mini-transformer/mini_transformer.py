"""Mini Transformer autoregresivo desde cero (arquitectura propia, PyTorch CPU)."""
import argparse
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

DIR = Path(__file__).resolve().parent
CORPUS = DIR.parent / "01-red-neuronal" / "datos" / "cuentos.txt"
PESOS = DIR / "mini_transformer.pt"

class MiniTransformer(nn.Module):
    def __init__(self, vocab, contexto=16, dimension=32, cabezas=2):
        super().__init__()
        if dimension % cabezas:
            raise ValueError("dimension debe ser múltiplo de cabezas")
        self.contexto = contexto
        self.dimension = dimension
        self.cabezas = cabezas
        self.tokens = nn.Embedding(vocab, dimension)
        self.posiciones = nn.Embedding(contexto, dimension)
        self.qkv = nn.Linear(dimension, 3 * dimension)
        self.proyeccion = nn.Linear(dimension, dimension)
        self.norma1 = nn.LayerNorm(dimension)
        self.norma2 = nn.LayerNorm(dimension)
        self.mlp = nn.Sequential(nn.Linear(dimension, 4 * dimension),
                                 nn.ReLU(), nn.Linear(4 * dimension, dimension))
        self.salida = nn.Linear(dimension, vocab)

    def forward(self, ids):
        B, T = ids.shape
        if T > self.contexto:
            raise ValueError("La secuencia excede el contexto")
        x = self.tokens(ids) + self.posiciones(torch.arange(T, device=ids.device))
        h = self.norma1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        def separar(t):
            return t.reshape(B, T, self.cabezas, self.dimension // self.cabezas).transpose(1, 2)
        q, k, v = map(separar, (q, k, v))
        scores = q @ k.transpose(-2, -1) / ((self.dimension // self.cabezas) ** .5)
        mascara = torch.triu(torch.ones(T, T, device=ids.device, dtype=torch.bool), diagonal=1)
        pesos = F.softmax(scores.masked_fill(mascara, float("-inf")), dim=-1)
        contexto = (pesos @ v).transpose(1, 2).contiguous().reshape(B, T, self.dimension)
        x = x + self.proyeccion(contexto)
        x = x + self.mlp(self.norma2(x))
        return self.salida(x)

def preparar():
    texto = CORPUS.read_text(encoding="utf-8").lower()
    vocab = sorted(set(texto))
    mapa = {c: i for i, c in enumerate(vocab)}
    return vocab, torch.tensor([mapa[c] for c in texto], dtype=torch.long)

def lote(ids, contexto, tam, generador):
    posiciones = torch.randint(len(ids) - contexto, (tam,), generator=generador)
    x = torch.stack([ids[i:i+contexto] for i in posiciones.tolist()])
    y = torch.stack([ids[i+1:i+contexto+1] for i in posiciones.tolist()])
    return x, y

def entrenar(args):
    torch.set_num_threads(1)
    torch.manual_seed(args.semilla)
    vocab, ids = preparar()
    corte = int(len(ids) * .8)
    if min(corte, len(ids)-corte) <= args.contexto:
        raise SystemExit("Corpus insuficiente para este contexto")
    train, valid = ids[:corte], ids[corte:]
    modelo = MiniTransformer(len(vocab), args.contexto, args.dimension, args.cabezas)
    optim = torch.optim.AdamW(modelo.parameters(), lr=args.tasa)
    generador = torch.Generator().manual_seed(args.semilla)
    print(f"Parámetros: {sum(p.numel() for p in modelo.parameters())}")
    for paso in range(1, args.pasos+1):
        modelo.train()
        x, y = lote(train, args.contexto, args.lote, generador)
        logits = modelo(x)
        loss = F.cross_entropy(logits.reshape(-1, len(vocab)), y.reshape(-1))
        optim.zero_grad()
        loss.backward()
        optim.step()
        if paso == 1 or paso % max(1, args.pasos//10) == 0:
            modelo.eval()
            with torch.no_grad():
                xv, yv = lote(valid, args.contexto, min(args.lote, 32), generador)
                val = F.cross_entropy(modelo(xv).reshape(-1, len(vocab)), yv.reshape(-1))
            print(f"paso {paso:4d} | train={loss.item():.3f} | valid={val.item():.3f}")
    torch.save({"estado":modelo.state_dict(), "vocab":vocab,
                "contexto":args.contexto, "dimension":args.dimension,
                "cabezas":args.cabezas}, PESOS)
    print(f"Guardado en {PESOS}")

def generar(args):
    torch.set_num_threads(1)
    if not PESOS.exists():
        raise SystemExit("Primero ejecutá entrenar")
    # weights_only evita deserialización arbitraria de objetos externos.
    datos = torch.load(PESOS, map_location="cpu", weights_only=True)
    vocab = datos["vocab"]
    mapa = {c:i for i,c in enumerate(vocab)}
    inicio = args.inicio.lower()
    if not inicio or any(c not in mapa for c in inicio):
        raise SystemExit("El inicio debe usar caracteres conocidos")
    modelo = MiniTransformer(len(vocab), datos["contexto"], datos["dimension"], datos["cabezas"])
    modelo.load_state_dict(datos["estado"])
    modelo.eval()
    torch.manual_seed(args.semilla)
    secuencia = torch.tensor([[mapa[c] for c in inicio]], dtype=torch.long)
    with torch.no_grad():
        for _ in range(args.longitud):
            logits = modelo(secuencia[:, -modelo.contexto:])[:, -1, :] / args.temperatura
            siguiente = torch.multinomial(F.softmax(logits, dim=-1), 1)
            secuencia = torch.cat([secuencia, siguiente], dim=1)
    print("".join(vocab[i] for i in secuencia[0].tolist()))

def main():
    parser = argparse.ArgumentParser(description="Laboratorio 04: mini Transformer")
    sub = parser.add_subparsers(dest="comando", required=True)
    tr = sub.add_parser("entrenar")
    tr.add_argument("--pasos", type=int, default=500)
    tr.add_argument("--contexto", type=int, default=16)
    tr.add_argument("--dimension", type=int, default=32)
    tr.add_argument("--cabezas", type=int, default=2)
    tr.add_argument("--lote", type=int, default=16)
    tr.add_argument("--tasa", type=float, default=.003)
    tr.add_argument("--semilla", type=int, default=42)
    ge = sub.add_parser("generar")
    ge.add_argument("--inicio", default="el ")
    ge.add_argument("--longitud", type=int, default=150)
    ge.add_argument("--temperatura", type=float, default=.8)
    ge.add_argument("--semilla", type=int, default=7)
    args = parser.parse_args()
    if args.comando == "entrenar":
        if min(args.pasos,args.contexto,args.dimension,args.cabezas,args.lote,args.tasa) <= 0:
            parser.error("Todos los parámetros deben ser positivos")
        if args.dimension % args.cabezas:
            parser.error("dimension debe ser divisible por cabezas")
        entrenar(args)
    else:
        if args.longitud < 0 or args.temperatura <= 0:
            parser.error("Longitud no negativa, temperatura positiva")
        generar(args)

if __name__ == "__main__":
    main()
