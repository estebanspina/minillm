# Laboratorio 05 — Experimentar, medir y explicar

**Pregunta:** ¿un contexto mayor produce mejores predicciones para este corpus?

Este laboratorio ejecuta el **mismo Mini Transformer** del laboratorio 04 con ventanas de distinta longitud, semilla fija y cantidad de pasos constante. Guarda un CSV para comparar la pérdida reportada.

## Ejecutar

Desde `05-experimentos/`, después de instalar las dependencias del laboratorio 04:

```bash
python experimentos.py --contextos 8 16 --pasos 100
python -m unittest discover -p "test_*.py"
```

Si tenés tiempo y CPU disponible, probá 300–500 pasos. El experimento hace corridas secuenciales y sobreescribe `04-mini-transformer/mini_transformer.pt` en cada una; **conservá una copia si querés guardar pesos**. El CSV `resultados.csv` se ignora en Git.

## Registro de investigación

Antes de ejecutar, completá:

- **Hipótesis:** predigo que…
- **Variable independiente:** número de caracteres de contexto.
- **Variables controladas:** corpus, dimensiones, cantidad de pasos, semilla y tasa de aprendizaje.
- **Métrica observada:** entropía cruzada de validación, menor es mejor.
- **Resultado:** compará las filas con el último paso de cada contexto.
- **Interpretación:** ¿la diferencia respalda la hipótesis? ¿Qué no demuestra?

## Advertencias metodológicas

1. El corpus es pequeño, repetido y sintético. No representa lenguaje natural.
2. El entrenamiento del laboratorio 04 utiliza lotes aleatorios; su validación también muestrea lotes. La métrica puede variar entre pasos, por lo que **un único resultado no alcanza para concluir**.
3. Mantener pasos constantes no iguala tiempo de cómputo ni cantidad de parámetros. Una ventana mayor cuesta más.
4. La partición temporal de este corpus repetitivo puede compartir frases entre entrenamiento y validación: no demuestra generalización fuera de distribución.
5. El CSV contiene valores redondeados por el entrenador (tres decimales).
6. Hacé múltiples corridas con diferentes semillas y reportá dispersión antes de afirmar que un modelo es mejor.

## Desafío de extensión

Agregá un comando para comparar 3 semillas y calcular promedio y desvío estándar de la pérdida final de validación. Después, repetí con otro corpus y explicá qué hipótesis cambiaron.
