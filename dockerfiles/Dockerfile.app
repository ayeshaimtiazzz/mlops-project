# ===============================
# Base image
# ===============================
FROM tensorflow/tensorflow:latest

# ===============================
# Set working directory
# ===============================
WORKDIR /app
# Copy model into container
COPY notebooks/models/emnist_federated_model.h5 /app/models/emnist_federated_model.h5

# ===============================
# Copy requirements and install
# ===============================
COPY pipeline/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# ===============================
# Copy Flask app code
# ===============================
COPY pipeline/app.py ./app.py
COPY pipeline/prometheus_metric.py ./prometheus_metric.py

# ===============================
# Expose Flask port
# ===============================
EXPOSE 5000

# ===============================
# Create folders for user data and models
# ===============================
RUN mkdir -p /data/user_drawings /models

# ===============================
# Command to run Flask app
# ===============================
CMD ["python", "app.py"]
