# Dockerfile content
# Use official lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire app code
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Run Streamlit app with no browser open and appropriate port
CMD ["streamlit", "run", "ui/streamlit_app.py", "--server.port=8501", "--server.enableCORS=false"]

