# Dockerfile for the F1 Strategy Simulator API (Optimized Production)

# Start from a lightweight, stable Python base image.
FROM python:3.11-slim

# Set the working directory inside the container.
WORKDIR /app

# Create a non-root user for security. This is a critical production practice.
RUN useradd --create-home --shell /bin/bash appuser

# Copy the requirements file.
COPY src/api/requirements.txt .

# --- THE FIX ---
# Install dependencies and then immediately clean up the pip cache in the same layer.
# This significantly reduces the final image size.
RUN pip install --no-cache-dir -r requirements.txt gunicorn && \
    rm -rf /root/.cache/pip

# Copy the application source code.
COPY src/api/ .

# Set ownership of the app directory to the non-root user.
RUN chown -R appuser:appuser /app

# Switch to the non-root user for runtime.
USER appuser

# Expose the port the app will run on.
EXPOSE 8080

# The command to run the application using Gunicorn.
CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:8080", "app:app"]