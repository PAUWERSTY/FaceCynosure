# === Stage 1: Base y Dependencias del Sistema ===

# Usar una imagen base de Python estable. 3.10 o 3.11 suelen ser buenas opciones.
# Usar 'slim' para reducir el tamaño de la imagen.
FROM python:3.10-slim as base

# Establecer variables de entorno importantes
# Evita prompts durante la instalación de paquetes apt
ENV DEBIAN_FRONTEND=noninteractive
# Asegura que los logs de Python salgan inmediatamente, crucial para debugging
ENV PYTHONUNBUFFERED=1
# Evita crear archivos .pyc (ahorra poco espacio, pero limpia)
ENV PYTHONDONTWRITEBYTECODE=1
# Deshabilita la caché de pip para mantener la imagen final más pequeña
ENV PIP_NO_CACHE_DIR=1

# Instalar dependencias del sistema operativo necesarias para
# OpenCV (cv2) y potencialmente otras librerías de ML/imagen.
# La lista exacta puede variar, pero esta cubre las más comunes.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
 # Limpiar caché de apt para reducir tamaño de la imagen
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# === Stage 2: Dependencias de Python ===

# Crear una nueva etapa o continuar en la misma. Usar una nueva es bueno si quieres separar.
# Vamos a continuar en la misma por simplicidad ahora.

# Establecer el directorio de trabajo DENTRO del contenedor
WORKDIR /app

# Copiar SOLO el archivo de requerimientos primero.
# Esto aprovecha el caché de Docker: si requirements.txt no cambia,
# Docker reutilizará la capa de instalación de pip, acelerando builds futuros.
COPY requirements.txt ./

# Actualizar pip e instalar las dependencias de Python listadas en requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# === Stage 3: Copiar Código y Ejecutar ===

# Copiar el resto del código de tu aplicación backend al directorio de trabajo /app
COPY . .

# Indicar a Docker que el contenedor escuchará en este puerto en tiempo de ejecución.
# Railway usará esto para mapear el tráfico externo.
EXPOSE 8000

# El comando por defecto para ejecutar cuando el contenedor inicie.
# Ejecuta Uvicorn con tu aplicación FastAPI (main.py -> app).
# Escucha en todas las interfaces (0.0.0.0) en el puerto 8000.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
