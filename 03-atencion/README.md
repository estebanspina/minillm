# Laboratorio 03 — Self-attention causal

**Objetivo:** comprender Q (queries), K (keys), V (values), puntajes, softmax y máscara causal con matrices pequeñas.

## Ejecutar

```bash
cd 03-atencion
python -m pip install -r ../01-red-neuronal/requirements.txt
python atencion.py --texto "hola"
python -m unittest discover -p "test_*.py"
```

La salida muestra una matriz triangular inferior: la fila i solo puede consultar columnas 0..i.

## Cómo funciona

1. Cada carácter obtiene un vector (embedding aleatorio).
2. Tres matrices proyectan embeddings en **Q**, **K** y **V**.
3. Los puntajes se calculan con `Q @ K.T / sqrt(d)`.
4. Una **máscara causal** asigna `-inf` a posiciones futuras.
5. `softmax` normaliza puntajes permitidos a probabilidades.
6. El promedio ponderado `pesos @ V` crea representaciones que incluyen contexto anterior.

**Precisión importante:** este laboratorio no entrena Q/K/V. Son matrices aleatorias para mostrar la mecánica. Todavía no produce lenguaje ni es un Transformer completo.

## Ejercicios

- Dibujá a mano la máscara causal para una secuencia de cuatro tokens.
- Comprobá que cada fila de atención suma 1.
- Modificá el último token de la secuencia: ¿cambia la salida del primero?
- Retirá temporalmente la máscara y compará los resultados.
- Explicá por qué cada token puede prestar atención a distintos tokens anteriores.

## Preguntas para compartir

¿En qué se diferencia la ventana fija del laboratorio 02 de la atención causal? ¿Qué deberían aprender las matrices Q, K y V si se entrenaran mediante gradientes?
