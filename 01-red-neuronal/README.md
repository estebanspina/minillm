# Laboratorio 01 — Tu primera red neuronal de lenguaje

**Tiempo sugerido:** 90 minutos. **Requisitos:** Python 3.10+, conocimientos básicos y NumPy. CPU, sin GPU ni modelos preentrenados.

## Objetivo
Construir un modelo neuronal que aprenda **P(carácter siguiente | carácter actual)**. No es un Transformer ni un LLM grande. Su falta de memoria contextual será el punto de partida de los siguientes laboratorios.

## Ejecutar
Desde esta carpeta:

```bash
python -m pip install -r requirements.txt
python minillm.py entrenar
python minillm.py generar --inicio "el " --longitud 200
```

En Windows podés usar `py` en lugar de `python`; en Linux/macOS, `python3`.

## Entender el código
1. `preparar_datos` convierte cada carácter en un entero (token).
2. `E[x]` busca un embedding entrenable del carácter anterior.
3. `E[x] @ W + b` calcula puntuaciones (logits) para el siguiente.
4. `softmax` convierte esas puntuaciones en probabilidades.
5. La pérdida es `-log(probabilidad del carácter correcto)`.
6. Los gradientes `dlogits`, `dW`, `db` y `dE` se calculan manualmente.
7. El descenso por gradiente actualiza `E`, `W` y `b`.
8. Durante la generación, se muestrea un carácter y se repite el proceso.

## Experimentos
```bash
python minillm.py entrenar --pasos 100
python minillm.py entrenar --pasos 3000 --dimension 32
python minillm.py generar --inicio "la " --temperatura 0.5
python minillm.py generar --inicio "la " --temperatura 1.5
```

**Importante:** cada nuevo entrenamiento sobrescribe `modelo.npz`. Hacé una copia si querés conservar los pesos de una corrida. El archivo de pesos está excluido de Git.

## Guía de clase (90 minutos)
- 0–15: predecir caracteres contando frecuencias a mano.
- 15–30: identificar tokens, embeddings, logits y probabilidades.
- 30–50: entrenar, registrar la pérdida y explicar los gradientes.
- 50–70: cambiar el corpus y experimentar con temperaturas.
- 70–90: comparar generaciones y explicar por qué no entiende oraciones.

## Preguntas de reflexión
- ¿Qué aprende el modelo y qué se programa explícitamente?
- ¿Por qué dos generaciones pueden ser diferentes?
- ¿Qué impide generar historias con coherencia?
- ¿Cómo agregarías varios caracteres de contexto en la siguiente etapa?

El corpus es **sintético**, corto y repetitivo, diseñado exclusivamente para una demostración educativa. No sirve para medir calidad real de lenguaje.
