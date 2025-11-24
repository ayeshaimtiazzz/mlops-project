# Use TensorFlow base image (includes Python, TensorFlow, and NumPy)
FROM tensorflow/tensorflow:latest-py3

# Set the working directory inside the container
WORKDIR /app

# Upgrade pip to the latest version (helps with dependency resolution)
RUN pip install --upgrade pip

# Copy requirements.txt and install dependencies (only non-conflicting ones)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code, custom module, and model
COPY app.py .
COPY prometheus_metric.py .
COPY emnist_federated_model.h5 .

# Expose the port the app runs on
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]