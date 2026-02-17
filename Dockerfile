FROM python:3.11-slim

WORKDIR /app

# Build arguments
ARG ENVIRONMENT=development

# Set environment variables
ENV ENVIRONMENT=${ENVIRONMENT}

# Copiar apenas os arquivos de requisitos primeiro para aproveitar o cache de camadas do Docker
COPY requirements.txt .

# Install git, dependencies, then cleanup
RUN apt-get update && \
    apt-get install -y git libgomp1 && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get purge -y git && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Install basic dependencies and the Microsoft ODBC 17 driver
RUN apt-get update && apt-get install -y gnupg2 curl ca-certificates unixodbc unixodbc-dev \
    && curl https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg \
    && curl https://packages.microsoft.com/config/debian/12/prod.list > /etc/apt/sources.list.d/mssql-release.list \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/microsoft-prod.gpg] https://packages.microsoft.com/debian/12/prod bookworm main" > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql17 mssql-tools \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copiar o restante dos arquivos da aplicação
COPY . .

# Expor a porta que o FastAPI usará
EXPOSE 80

# Comando para iniciar a aplicação
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
