# Instalación local — Windows y Linux

El proyecto utiliza Python y funciona en CPU. **No utiliza Ollama ni descarga modelos preentrenados.**

## Requisitos

- Python 3.10 o superior, terminal y editor de texto.
- Laboratorio 00: biblioteca estándar, sin instalaciones adicionales.
- Laboratorios 01–03 y 05: NumPy (el 05 también requiere PyTorch).
- Laboratorio 04 y experimentos del 05: PyTorch en CPU; su instalación puede consumir bastante espacio en disco y requiere descarga inicial.

## Windows (PowerShell)

```powershell
git clone https://github.com/estebanspina/minillm.git
cd minillm
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r 01-red-neuronal/requirements.txt
python 00-bigramas/bigramas.py
python 01-red-neuronal/minillm.py entrenar --pasos 100
```

Si PowerShell no permite activar el entorno, podés usar directamente `.venv\Scripts\python.exe` en lugar de activar o abrir una consola CMD.

Para ejecutar el Transformer:

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python 04-mini-transformer/mini_transformer.py entrenar --pasos 50
```

## Linux (Bash)

```bash
git clone https://github.com/estebanspina/minillm.git
cd minillm
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r 01-red-neuronal/requirements.txt
python 00-bigramas/bigramas.py
python 01-red-neuronal/minillm.py entrenar --pasos 100
```

Para el Transformer:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python 04-mini-transformer/mini_transformer.py entrenar --pasos 50
```

En distribuciones Debian/Ubuntu puede ser necesario instalar `python3-venv` mediante el administrador de paquetes.

## Pruebas locales

Cada laboratorio tiene sus propias instrucciones. Las pruebas se ejecutan desde **su carpeta**:

```bash
cd 01-red-neuronal
python -m unittest discover -p "test_*.py"
```

Repetí para los laboratorios 00, 02, 03 y 04 según sus README. El laboratorio 05 también cuenta con pruebas.

## Equipos de pocos recursos

- Empezá por el laboratorio 00 sin instalaciones.
- Para los laboratorios 01–03, instalá solo NumPy.
- Usá pocos pasos (`--pasos 10`) para comprobar que todo funciona.
- Evitá instalar PyTorch en clase si la conexión es limitada: prepará las dependencias de antemano.
- No prometemos tiempos exactos de entrenamiento; dependen de la CPU, RAM y el sistema operativo.
- Para revisar errores de descarga o instalación, registrá la versión de Python y el comando que falló.

## Sin Git

Podés descargar el ZIP del repositorio desde GitHub > Code > Download ZIP y abrir una terminal en la carpeta extraída.
