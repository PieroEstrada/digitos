# Usa una imagen base de Python
FROM python:3.11-slim
# Establece el directorio de trabajo
# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

COPY requirements.txt .


RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

# Comando para ejecutar la aplicación
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}
