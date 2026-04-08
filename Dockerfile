FROM python:3.10-slim

# Set up a new user named "user" with user ID 1000 per Hugging Face Spaces requirements
RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Copy the current directory contents into the container, setting the owner to the user
COPY --chown=user . $HOME/app

# Install dependencies directly
RUN pip install --no-cache-dir openenv-core pydantic openai fastapi uvicorn

# Ensure the server path is recognizable as a package
ENV PYTHONPATH=$HOME/app

EXPOSE 7860

# Start FastAPI app
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
