"""Ejecuta experimentos comparables del laboratorio 04 y guarda métricas en CSV."""
import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
MODELO = BASE.parent / "04-mini-transformer" / "mini_transformer.py"
RESULTADOS = BASE / "resultados.csv"
PATRON = re.compile(r"paso\s+(\d+)\s+\|\s+train=([\d.]+)\s+\|\s+valid=([\d.]+)")

def extraer_metricas(salida):
    """Parsea las métricas reportadas por el entrenador."""
    return [(int(n), float(train), float(valid)) for n, train, valid in PATRON.findall(salida)]

def ejecutar(contexto, pasos, semilla, dimension):
    comando = [sys.executable, str(MODELO), "entrenar",
               "--contexto", str(contexto), "--pasos", str(pasos),
               "--dimension", str(dimension), "--semilla", str(semilla)]
    resultado = subprocess.run(comando, text=True, capture_output=True, check=True)
    metricas = extraer_metricas(resultado.stdout)
    if not metricas:
        raise RuntimeError("El entrenamiento no produjo métricas reconocibles")
    return metricas

def main():
    p = argparse.ArgumentParser(description="Comparar tamaño de contexto del Mini Transformer")
    p.add_argument("--pasos", type=int, default=100)
    p.add_argument("--contextos", type=int, nargs="+", default=[8, 16])
    p.add_argument("--dimension", type=int, default=32)
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--salida", type=Path, default=RESULTADOS)
    args = p.parse_args()
    if args.pasos < 1 or args.dimension < 1 or any(n < 1 for n in args.contextos):
        p.error("Pasos, dimensión y contextos deben ser positivos")
    # El modelo por defecto tiene dos cabezas.
    if args.dimension % 2:
        p.error("La dimensión debe ser múltiplo de 2")
    filas = []
    for contexto in args.contextos:
        print(f"Entrenando contexto={contexto}, pasos={args.pasos}...", flush=True)
        for paso, train, valid in ejecutar(contexto, args.pasos, args.semilla, args.dimension):
            filas.append({"contexto": contexto, "paso": paso, "train": train,
                          "valid": valid, "semilla": args.semilla, "dimension": args.dimension})
        print(f"Última pérdida validación: {filas[-1]['valid']:.3f}")
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    with args.salida.open("w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=["contexto", "paso", "train", "valid", "semilla", "dimension"])
        escritor.writeheader()
        escritor.writerows(filas)
    print(f"Datos guardados en: {args.salida}")

if __name__ == "__main__":
    main()
