FROM python:3.9-slim

WORKDIR /app

# Install git for pip to clone from GitHub
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copiar apenas os arquivos de requisitos primeiro para aproveitar o cache de camadas do Docker
COPY requirements.txt .

# Replace token placeholder with build arg
ARG GH_TOKEN
RUN sed -i "s|<GH_TOKEN>|${GH_TOKEN}|g" requirements.txt

# Instalar dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o restante dos arquivos da aplicação
COPY . .

# Expor a porta que o FastAPI usará
EXPOSE 80

# Comando para iniciar a aplicação
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
