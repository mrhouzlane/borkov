FROM python:3.13-slim

WORKDIR /app

# Install system tools
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv globally
RUN pip install --upgrade pip && pip install uv

# Copy project files (including .env if you want local env config baked-in)
COPY . .

# Create virtual environment and sync dependencies.
RUN uv venv && uv pip install --upgrade pip && uv sync --no-dev

# Expose agent port(s) if agent/server
EXPOSE 3001 5555

# By default, run your agent.
# MAIN THING: The way CMD is listed runs 'uv' itself as an executable, not as 'python main.py'.
CMD ["uv", "run", "main.py"]
