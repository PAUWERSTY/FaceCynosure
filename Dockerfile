# 
5.  **Entorno Virtual (Opcional pero Recomendado):** No estás usando un entorno virtual dentro del Dockerfile. Aunque no es estrictamente necesario en1. Imagen Base: Usar una versión ligeramente más probada como 3.11 un contenedor (que ya es aislado), usarlo puede a veces evitar conflictos con paquetes en lugar de 3.12
#    puede evitar algunos problemas de compatibilidad inicial del sistema si usaras `--system-site-packages` (que no eses con librerías complejas.
#    Si necesitas 3.12 específicamente el caso aquí, pero es buena práctica).

**Dockerfile Mejorado:**

```dockerfile
# Us, puedes volver a cambiarlo, pero 3.11 es muy estable.
FROM python:3.11-slim
# FROM python:3.12ar una versión de Python estable como 3.11 o 3.10 si 3.12 da problemas
# FROM python:3.12-slim # Descomenta esta y comenta la anterior si prefieres 3.12

-slim
FROM python:3.11-slim
# FROM python:3.10# 2. Establecer Variables de Entorno para Python
#    --slim

# Establecer variables de entorno para evitar preguntas interactivas durante apt-get
ENV DEBIAN_FRONTEND=noninteractive \
    # PYTHONUNBUFFERED: Asegura que los logs de Python aparezcan inmediatamente.
#    - PYTHONDONTWRITEBYTECODE: Evita crear archivos .pyc, Opcional: Variables para configurar pip
    PIP_NO_CACHE_DIR= útil en contenedores.
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
#    - Opcional: Especificar exploff \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIPícitamente la codificación puede ayudar en algunos casos
# ENV LANG C_DEFAULT_TIMEOUT=100 \
    # Opcional: Variables.UTF-8
# ENV LC_ALL C.UTF-8

# 3. Instalar Dependencias del Sistema Operativo (MÁS COMPLETAS)
#     de entorno de Python
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random

# EstableOpenCV y a veces TensorFlow/Torch pueden necesitar más que solo libgl1 y libcer el directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema operativoglib.
#    Añadir build-essential si algo necesita compilarse, y otras REQUERIDAS y recomendadas
# Se añaden build-essential, cmake libs comunes.
RUN apt-get update && apt-get install -y --no-install-re (a veces necesario), libsm6, libxext6, libxrender-dev
RUN apt-commends \
    build-essential \
    cmake \
    pkgget update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    -config \
    # Dependencias comunes para OpenCV GUI/headless
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1libxrender-dev \
    # Limpiar caché de apt para reducir tamaño de imagen
    && apt-get clean \
    && rm -rf /var \
    # Dependencias adicionales a veces necesarias por librerías de ML/lib/apt/lists/*

# Copiar solo el archivo de requer
    libopenblas-dev \
    liblapack-dev \
    # Limpiar caché de apt para reducir tamaño de imagen
    && apt-get clean \
    && rmimientos primero (aprovecha caché de Docker)
COPY requirements.txt .

# Actualizar pip -rf /var/lib/apt/lists/*

# 4. Establecer Directorio de Trabajo
WORKDIR /app

# 5. Cop e instalar dependencias de Python desde requirements.txt
# Usar caché de montaje siiar SOLO requirements.txt primero (para caché de Docker)
COPY requirements.txt ./

 se despliega en plataformas que lo soportan como Railway
# RUN --mount=type=cache,target=/root/.cache/pip \
#    pip install --upgrade pip && \# 6. Instalar Dependencias de Python
#    - Actualizar pip primero.
#    - Usar --no-cache-dir para evitar
#    pip install -r requirements.txt
# Si no usas caché llenar el caché dentro de la capa.
#    - Railway maneja el caché de montaje (o para pruebas locales):
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copiar el resto del código de la aplicación externo con --mount, así que no necesitamos duplicarlo aquí
#      si Railway
# Asegúrate de tener un .dockerignore para excluir node_modules, venv, etc.
COPY . .

# Exponer el puerto interno en lo añade automáticamente al comando RUN. Si no, añadirías:
#      --mount=type=cache,target=/root/.cache/pip
RUN pip install --no-cache-dir -- el que correrá Uvicorn
EXPOSE 8000

# Comando para ejecutar la aplicación
# Usar el puerto 8000 (Railway lo mapeará)
CMD ["uvicorn", "main:app", "--host", "0.upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 7. Copiar el Resto del Código de la Aplicación
0.0.0", "--port", "8000"]
