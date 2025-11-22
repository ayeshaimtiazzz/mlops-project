# evaluation.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import load_model

# ===============================
# HYPERPARAMETERS
# ===============================
BATCH_SIZE = 32

# ===============================
# LOAD MODEL
# ===============================
model_path = "//mnt//c//Users//aimte//OneDrive//Desktop//uni work//mlops//project_mlops//model//emnist_model_1.h5"
model = load_model(model_path)
print(f"Loaded model from {model_path}")

# ===============================
# LOAD EMNIST TEST DATA
# ===============================
# Using TFF simulation dataset
import tensorflow_federated as tff
_, test_source = tff.simulation.datasets.emnist.load_data()

def preprocess_test(example):
    return tf.reshape(example['pixels'], [-1]), example['label']

test_ds = test_source.create_tf_dataset_from_all_clients().map(preprocess_test).batch(BATCH_SIZE)

# Collect test images and labels
test_images, test_labels = [], []
for images, labels in test_ds.take(50):  # take some batches to speed up
    test_images.append(images.numpy())
    test_labels.append(labels.numpy())

test_images = np.concatenate(test_images)
test_labels = np.concatenate(test_labels)

# ===============================
# EVALUATE MODEL
# ===============================
loss, accuracy = model.evaluate(test_images, test_labels, batch_size=BATCH_SIZE, verbose=0)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# ===============================
# PREDICTIONS AND METRICS
# ===============================
predictions = np.argmax(model.predict(test_images, verbose=0), axis=1)

# Confusion matrix
cm = confusion_matrix(test_labels, predictions)
print("Confusion Matrix (subset):")
print(cm[:10,:10])  # print first 10x10 for readability

# Classification report
print("\nClassification Report:")
print(classification_report(test_labels, predictions, zero_division=0))

# ===============================
# VISUALIZATIONS
# ===============================

# Confusion matrix heatmap
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()
print("Confusion matrix saved as 'confusion_matrix.png'")

# Sample predictions visualization
plt.figure(figsize=(12,6))
for i in range(12):
    plt.subplot(3,4,i+1)
    plt.imshow(test_images[i].reshape(28,28), cmap='gray')
    plt.title(f"True:{test_labels[i]}, Pred:{predictions[i]}")
    plt.axis('off')
plt.tight_layout()
plt.savefig('sample_predictions.png')
plt.show()
print("Sample predictions saved as 'sample_predictions.png'")

# ===============================
# DONE
# ===============================
print("\n✅ Evaluation complete!")
