FROM python:3.11.6-slim

# Install uv.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy the application into the container.
COPY . /app

# Install the application dependencies.
WORKDIR /app
RUN uv sync --frozen --no-cache

EXPOSE 9000

# Run the application.
# CMD ["/app/.venv/bin/fastapi", "run", "main.py", "--port", "9000", "--host", "0.0.0.0"]
CMD ["uv", "run", "uvicorn", "main:app", "--port", "9000", "--host", "0.0.0.0", "--timeout-keep-alive", "60"]
