# Use secure base image
FROM python:3.10-slim-bullseye

# Metadata labels
LABEL maintainer="YourName"
LABEL description="Secure Fetal Health App (Flask + Streamlit)"

# Set working directory
WORKDIR /app

# Copy all files into container
COPY . .

# Update + secure base system
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends gcc && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get purge -y gcc && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Expose Streamlit and Flask ports
EXPOSE 8501
EXPOSE 5000

# Optional: set non-root user for container security
RUN adduser --disabled-password --gecos '' appuser && chown -R appuser /app
USER appuser

# Use APP environment variable to toggle Flask or Streamlit
CMD ["sh", "-c", "if [ \"$APP\" = 'flask' ]; then python flaskapp.py; else streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0; fi"]

