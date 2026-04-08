FROM python:3.10-slim

WORKDIR /app

# Copy project files
COPY . .

# Install dependencies directly
RUN pip install --no-cache-dir openenv-core pydantic openai fastapi uvicorn

# Ensure the server path is recognizable as a package (optional if pythonpath is set)
ENV PYTHONPATH=/app

EXPOSE 8000

# Start FastAPI app
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
