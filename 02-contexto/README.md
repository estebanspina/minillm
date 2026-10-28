# Laboratorio 02 — Predicción con contexto

**Objetivo:** predecir el siguiente carácter con los últimos **4 caracteres** (configurable), en lugar de mirar solo uno.

## Requisitos
Python 3.10+, NumPy, CPU. Utiliza el corpus del Laboratorio 01.

```bash
cd 02-contexto
python -m pip install -r ../01-red-neuronal/requirements.txt
python minillm_contexto.py entrenar --pasos 1500 --contexto 4
python minillm_contexto.py generar --inicio "el " --longitud 200
python -m unittest discover -p "test_*.py"
```

## Modelo
1. Construí ventanas de N caracteres: `"hola" -> " "` (ejemplo).
2. Buscá el embedding de cada carácter y concatenalos.
3. Aplicá una capa lineal con `tanh`, seguida por otra capa y `softmax`.
4. Medí pérdida de entropía cruzada y calculá gradientes a mano.
5. Reservá el último 20% del corpus como validación.
6. Guardá los pesos y generá un carácter por vez.

La ventana es un contexto **de tamaño fijo**, no self-attention. Es una MLP pequeña y todavía no es un Transformer.

## Experimentos
- Entrená con `--contexto 1`, `--contexto 2`, `--contexto 4` y `--contexto 8`.
- Compará pérdida **de validación**, no solo la de entrenamiento.
- Probá `--dimension 4` y `--dimension 16`.
- Contrastá salidas con el laboratorio 01.
- Explicá por qué duplicar el contexto aumenta los parámetros de `W1`.

**Advertencia:** el corpus es deliberadamente pequeño y repetitivo: buenas pérdidas no implican comprensión. El modelo de validación comparte distribución y fragmentos repetidos, por lo que no mide generalización real.
