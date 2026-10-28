# Guía para mentees — Aprender haciendo

Vas a construir modelos de lenguaje desde cero, de menor a mayor complejidad. Vas a entrenar pesos locales y generar texto. No vas a usar un chatbot, servicios externos ni pesos preentrenados.

## Tu bitácora

Para cada práctica, registrá:

1. ¿Qué predice este modelo?
2. ¿Qué información recibe para hacer su predicción?
3. ¿Qué parámetros cambian o qué frecuencias cuenta?
4. ¿Cómo comprobás que funciona?
5. ¿Qué no puede hacer?

## Desafíos por nivel

| Nivel | Desafío | Entrega |
| --- | --- | --- |
| Inicial | Modificar texto inicial y semilla en Lab 00 | Dos salidas y una explicación |
| Inicial | Cambiar pasos de entrenamiento en Lab 01 | Tabla de pérdidas |
| Intermedio | Comparar contextos 2, 4 y 8 en Lab 02 | Hipótesis y resultados |
| Intermedio | Explicar fila por fila la máscara de Lab 03 | Dibujo o matriz anotada |
| Avanzado | Cambiar cabezas y dimensiones en Lab 04 | Tabla de parámetros y pérdidas |
| Avanzado | Comparar varias semillas en Lab 05 | Informe reproducible y limitaciones |

## Cómo entregar

Creá una rama para tu ejercicio, documentá el experimento en Markdown y abrí un Pull Request. Incluí el comando ejecutado, la evidencia y lo que no funcionó. No subas pesos del modelo, datos personales ni corpus de terceros sin permiso.

## Glosario mínimo

- **Token:** unidad representada por un identificador numérico; en este proyecto, carácter.
- **Embedding:** vector entrenable de un token.
- **Logit:** puntuación previa a convertirla en probabilidad.
- **Softmax:** transforma logits en probabilidades.
- **Pérdida:** mide la diferencia entre lo predicho y lo correcto.
- **Gradiente:** indica cómo ajustar un parámetro para reducir pérdida.
- **Atención causal:** permite consultar el pasado y el presente, no posiciones futuras.
- **Sobreajuste:** memorizar patrones de entrenamiento sin mejorar fuera de esos datos.
