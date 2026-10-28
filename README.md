# MiniLLM

Un laboratorio educativo para construir modelos de lenguaje **desde cero** con Python, empezando por redes neuronales pequeñas y avanzando hacia la atención y los Transformers.

**Destinatarios:** estudiantes de 18 años o más con Python básico. **Objetivo:** funcionar localmente en CPU, sin Ollama, APIs ni pesos preentrenados.

## Recorrido

| Etapa | Qué construimos | Conceptos |
| --- | --- | --- |
| 00 | Modelo estadístico de caracteres | Tokens, conteos, probabilidades |
| 01 | Modelo neuronal de caracteres | Embeddings, softmax, pérdida, gradientes |
| 02 | Predicción con contexto | Secuencias, entrenamiento, evaluación |
| 03 | Mecanismo de atención | Self-attention y causalidad |
| 04 | Mini Transformer | Bloques, posiciones, generación |
| 05 | Experimentos | Generalización, límites y comparación |

Cada etapa tendrá instrucciones, código independiente y desafíos. **No buscamos crear un chatbot competitivo**: buscamos entender cómo aprende un modelo de lenguaje.

## Comenzar

Empezá con [la instalación en Windows o Linux](docs/INSTALACION.md). Después realizá los laboratorios en orden: [00](00-bigramas/README.md), [01](01-red-neuronal/README.md), [02](02-contexto/README.md), [03](03-atencion/README.md), [04](04-mini-transformer/README.md) y [05](05-experimentos/README.md).

Para organizar clases, consultá la [guía docente](docs/GUIA_DOCENTE.md). Si participás como estudiante, usá la [guía para mentees](docs/GUIA_MENTEE.md).

## Filosofía

- Código legible antes que sofisticación.
- Entrenamiento local y reproducible.
- Cero modelos preentrenados.
- Aprender también de los errores y de las limitaciones.

## Contribuciones

Trabajamos con issues y pull requests. Cada cambio debe incluir una explicación de qué se aprende y cómo verificarlo.
