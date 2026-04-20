# Use an official lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the default Streamlit port
EXPOSE 8501

# Healthcheck to ensure the container is running
HEALTHCHECK CMD python -c "import urllib.request, sys; sys.exit(0) if urllib.request.urlopen('http://localhost:8501/_stcore/health').status == 200 else sys.exit(1)"

# Run the application
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
