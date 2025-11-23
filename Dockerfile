FROM python:3.9-slim

WORKDIR /app

# Build arguments
ARG GH_TOKEN
ARG ENVIRONMENT=development

# Set environment variables
ENV ENVIRONMENT=${ENVIRONMENT}

# Copiar apenas os arquivos de requisitos primeiro para aproveitar o cache de camadas do Docker
COPY requirements.txt .

# Replace token placeholder with build arg
RUN sed -i "s|<GH_TOKEN>|${GH_TOKEN}|g" requirements.txt

# Install git, install dependencies, then remove git to keep image small
RUN apt-get update && \
    apt-get install -y git && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get purge -y git && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Copiar o restante dos arquivos da aplicação
COPY . .

# Expor a porta que o FastAPI usará
EXPOSE 80

# Comando para iniciar a aplicação
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
