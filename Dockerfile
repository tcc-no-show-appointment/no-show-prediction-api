FROM python:3.11-slim

WORKDIR /app

# Build arguments
ARG GH_TOKEN
ARG ENVIRONMENT=development

# Set environment variables
ENV ENVIRONMENT=${ENVIRONMENT}

# Copiar apenas os arquivos de requisitos primeiro para aproveitar o cache de camadas do Docker
COPY requirements.txt .

# Install git, configure credentials, install dependencies, then cleanup
RUN --mount=type=secret,id=gh_token \
    apt-get update && \
    apt-get install -y git && \
    GH_TOKEN=$(cat /run/secrets/gh_token 2>/dev/null || echo "${GH_TOKEN}") && \
    git config --global url."https://${GH_TOKEN}@github.com/".insteadOf "https://github.com/" && \
    pip install --no-cache-dir -r requirements.txt && \
    git config --global --unset url."https://${GH_TOKEN}@github.com/".insteadOf && \
    apt-get purge -y git && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Copiar o restante dos arquivos da aplicação
COPY . .

# Expor a porta que o FastAPI usará
EXPOSE 80

# Comando para iniciar a aplicação
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
