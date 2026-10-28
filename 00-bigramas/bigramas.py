"""Generador estadístico de caracteres. Solo Python estándar."""
import argparse
import random
from collections import defaultdict
from pathlib import Path

CORPUS = Path(__file__).resolve().parents[1] / "01-red-neuronal" / "datos" / "cuentos.txt"

def aprender(texto):
    """Cuenta qué carácter sigue a cada carácter."""
    conteos = defaultdict(lambda: defaultdict(int))
    for actual, siguiente in zip(texto, texto[1:]):
        conteos[actual][siguiente] += 1
    return {c: dict(siguientes) for c, siguientes in conteos.items()}

def distribucion(conteos, caracter):
    """Devuelve P(siguiente | caracter), en orden descendente."""
    opciones = conteos.get(caracter, {})
    total = sum(opciones.values())
    return sorted(((c, n / total) for c, n in opciones.items()), key=lambda p: -p[1]) if total else []

def generar(conteos, inicio, longitud=150, semilla=42):
    rng = random.Random(semilla)
    resultado = list(inicio)
    actual = inicio[-1]
    for _ in range(longitud):
        opciones = conteos.get(actual)
        if not opciones:
            break
        letras, pesos = zip(*opciones.items())
        actual = rng.choices(letras, weights=pesos, k=1)[0]
        resultado.append(actual)
    return "".join(resultado)

def main():
    parser = argparse.ArgumentParser(description="Laboratorio 00: bigramas")
    parser.add_argument("--inicio", default="el ")
    parser.add_argument("--longitud", type=int, default=150)
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--caracter", default="a", help="Muestra las probabilidades siguientes")
    args = parser.parse_args()
    if not args.inicio or args.longitud < 0 or len(args.caracter) != 1:
        parser.error("Usá inicio no vacío, longitud >= 0 y exactamente un caracter")
    texto = CORPUS.read_text(encoding="utf-8").lower()
    conteos = aprender(texto)
    print(f"Corpus: {len(texto)} caracteres. Caracteres distintos: {len(set(texto))}")
    print(f"Siguientes a {args.caracter!r}: {distribucion(conteos, args.caracter)[:10]}")
    print("Generación:")
    print(generar(conteos, args.inicio.lower(), args.longitud, args.semilla))

if __name__ == "__main__":
    main()
