# Guía docente — MiniLLM en seis clases

**Público:** mentees mayores de 18 años con Python básico. **Duración:** 6 clases de 90 minutos (ajustable). **Modalidad:** trabajo en parejas, CPU local. **Meta:** explicar cómo se aprende a predecir el siguiente carácter, construir un Transformer pequeño y reconocer sus límites.

## Antes de comenzar

1. Revisá [Instalación](INSTALACION.md) y probá los scripts en una laptop semejante a las del grupo.
2. Prepará un corpus sintético y evitá datos personales o textos sin autorización.
3. Definí parejas: una persona explica y otra ejecuta; cambian roles.
4. Pedí registrar hipótesis, comandos, pérdida y observaciones; **no premiar texto coherente como único indicador de aprendizaje**.

## Secuencia de clases

| Clase | Laboratorio | Pregunta guía | Evidencia |
| --- | --- | --- | --- |
| 1 | 00 — Bigramas | ¿Se puede generar texto contando letras? | Probabilidades y muestra generada |
| 2 | 01 — Red neuronal | ¿Qué cambia cuando el modelo aprende parámetros? | Curva o tabla de pérdida |
| 3 | 02 — Contexto | ¿Por qué sirve recordar más de un carácter? | Comparación de tamaños de ventana |
| 4 | 03 — Atención | ¿Cómo evitar mirar tokens del futuro? | Matriz causal y test explicado |
| 5 | 04 — Transformer | ¿Cómo se conectan los componentes? | Entrenamiento y generación en CPU |
| 6 | 05 — Experimentos | ¿Cómo comparar modelos sin engañarnos? | Informe de hipótesis y límites |

## Estructura sugerida por clase (90 min)

- 0–15: discusión y predicción de resultados.
- 15–30: lectura guiada del código.
- 30–60: implementación y ejecución en parejas.
- 60–75: cambio controlado de un parámetro.
- 75–90: explicación oral, evidencia y reflexión sobre límites.

## Evaluación formativa

| Criterio | Básico | Logrado | Destacado |
| --- | --- | --- | --- |
| Explicación | Identifica funciones | Explica datos y probabilidades | Justifica el flujo completo |
| Ejecución | Sigue pasos | Soluciona errores sencillos | Diseña una prueba adicional |
| Método | Observa una salida | Registra controles y métricas | Reconoce sesgos y sobreajuste |
| Colaboración | Participa | Comparte decisiones | Ayuda a otro equipo a explicar |

## Cierre

Cada pareja entrega un README breve con: hipótesis, comando reproducible, evidencia, interpretación y dos limitaciones de su modelo. El proyecto no intenta fabricar un chatbot inteligente: intenta comprender sus componentes fundamentales.
