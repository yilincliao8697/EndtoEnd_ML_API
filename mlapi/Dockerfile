FROM python:3.11-slim AS build
WORKDIR /app

# Install curl
RUN apt-get update \
    && apt-get install -y \
         curl \
         build-essential \
         libffi-dev \
    && rm -rf /var/lib/apt/lists/*

#Install Poetry
ENV POETRY_VERSION=1.8.3
ENV POETRY_HOME=/opt/poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH ${POETRY_HOME}/bin:${PATH}

# Create a Venv
RUN python -m venv --copies /venv

COPY pyproject.toml poetry.lock ./
RUN . /venv/bin/activate && poetry install --only main

# Run Image
FROM python:3.11-slim AS runner
WORKDIR /app

COPY --from=build /venv /venv
ENV PATH /venv/bin:${PATH}

# Copy source
COPY . .

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 CMD curl -f http://localhost:8000/health || exit 1

# Run
CMD ["/venv/bin/uvicorn", "src.main:app", "--host", "0.0.0.0"]