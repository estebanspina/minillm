# Laboratorio 00 — Generar texto sin redes neuronales

**Duración:** 60–90 minutos. **Dependencias:** ninguna; Python 3.10+.

## Qué vamos a aprender
Un modelo de lenguaje puede estimar la probabilidad del siguiente carácter a partir de ejemplos. Primero lo hacemos contando pares de caracteres, o *bigramas*, sin aprendizaje profundo.

## Ejecutar

Desde la carpeta `00-bigramas`:

```bash
python bigramas.py
python bigramas.py --inicio "la " --longitud 200 --semilla 12 --caracter l
python -m unittest discover -p "test_*.py"
```

El corpus compartido se encuentra en `../01-red-neuronal/datos/cuentos.txt`. Todo corre localmente sin instalar paquetes.

## Actividad guiada
1. Escribí `"casa"` y contá cuántas veces aparece cada par de letras.
2. Comprobá que las probabilidades que salen de una letra suman 1.
3. Explorá `aprender`, `distribucion` y `generar`.
4. Corré el mismo modelo con dos semillas distintas.
5. Editá el corpus y compará los resultados.

## Desafíos
- Mostrá los cinco caracteres más probables después de un espacio.
- Detectá el caso en que una letra nunca apareció antes de otra.
- Agregá un argumento para elegir otro archivo de entrenamiento.
- Compará este modelo con `../01-red-neuronal/minillm.py`: ¿qué se cuenta y qué se aprende mediante gradientes?

**Limitación deliberada:** el modelo solo observa el carácter actual. No comprende palabras, frases ni significados.
