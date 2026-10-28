# Laboratorio 04 — Un Transformer que aprende

Construimos **desde cero la arquitectura** de un Transformer autoregresivo de un bloque, sin descargar pesos preentrenados. Usamos PyTorch únicamente para tensores, diferenciación automática y optimización; la atención, máscara causal, residual, normalizaciones, embeddings y MLP están implementados en el código.

## Instalación (CPU, sin GPU)
Python 3.10+ y PyTorch. En una terminal dentro de `04-mini-transformer/`:

```bash
python -m pip install -r requirements.txt
python mini_transformer.py entrenar --pasos 500
python mini_transformer.py generar --inicio "el " --longitud 150
python -m unittest discover -p 'test_*.py'
```

La descarga inicial de PyTorch **sí requiere Internet** si no está instalado. Una vez instalado, entrenar y generar es completamente local. Para equipos con recursos muy limitados, los laboratorios 00–03 requieren menos memoria.

## Componentes
1. Embeddings de caracteres y posiciones.
2. Proyección lineal a Queries, Keys y Values.
3. Atención multi-cabeza causal: los tokens no pueden ver el futuro.
4. Conexión residual, normalización y MLP.
5. Predicción del siguiente carácter en **cada posición**.
6. Cross-entropy, retropropagación con autograd y AdamW.
7. Generación autoregresiva con temperatura.

## Actividades
- Compará `--contexto 8` y `--contexto 16`.
- Cambiá `--cabezas 1` y `--cabezas 2` (la dimensión debe ser divisible).
- Observá pérdida de entrenamiento y validación: ¿cuándo aparece sobreajuste?
- Explicá cómo el test `test_no_mira_el_futuro` detecta una falla en la máscara causal.

## Límites
Este es un **mini Transformer**, no un LLM capaz de conversar. El corpus sintético es minúsculo y repetitivo; sirve para enseñar mecanismos, no para evaluar comprensión o rendimiento lingüístico. Las métricas de validación son ilustrativas, no una prueba rigurosa de generalización.
