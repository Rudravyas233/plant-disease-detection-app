FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all files
COPY . .

# Install system dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for Streamlit
EXPOSE 10000

# Start Streamlit app
CMD ["streamlit", "run", "main.py", "--server.port=10000", "--server.address=0.0.0.0"]
