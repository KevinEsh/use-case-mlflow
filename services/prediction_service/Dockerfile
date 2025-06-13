# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Install the project into `/app`
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Install the project's dependencies using the lockfile and settings
# 1. Usa una caché persistente para los paquetes descargados por uv, acelerando builds futuros.
# 2. Monta el archivo uv.lock (lista exacta de dependencias) en el contenedor. Así, uv instala exactamente las versiones bloqueadas.
# 3. Monta el archivo pyproject.toml (lista de dependencias) en el contenedor. Así, uv puede resolver las dependencias correctamente.
# 4. --locked: Usa solo las versiones del lockfile.
# --no-install-project: No instala el propio proyecto, solo las dependencias.
# --no-dev: No instala dependencias de desarrollo.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

# Then, add the rest of the project source code and install it
# Installing separately from its dependencies allows optimal layer caching
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# Place executables in the environment at the front of the path
# TODO: - **Mejora posible:**  
#   Si tus servicios están en la misma red de Docker Compose, **mejor usa los nombres de servicio** en vez de la IP del host:
#   ```dockerfile
ENV PATH="/app/.venv/bin:$PATH"
ENV MLFLOW_S3_ENDPOINT_URL=http://172.17.0.1:9000
ENV MLFLOW_TRACKING_URI=http://172.17.0.1:5000
# ENV MLFLOW_S3_BUCKET=mlflowdata
ENV AWS_ACCESS_KEY_ID=minioadmin
ENV AWS_SECRET_ACCESS_KEY=minioadmin

# Reset the entrypoint, don't invoke `uv`
# Elimina cualquier entrypoint heredado de la imagen base.
ENTRYPOINT [] 

# Expose port for FastAPI
EXPOSE 8000

# Run the FastAPI application by default
# Uses `fastapi dev` to enable hot-reloading when the `watch` sync occurs
# Uses `--host 0.0.0.0` to allow access from outside the container
CMD ["fastapi", "dev", "--host", "0.0.0.0", "main.py"]

# Start FastAPI app
# CMD ["uvicorn", "app.main:api", "--host", "0.0.0.0", "--port", "8000"]

