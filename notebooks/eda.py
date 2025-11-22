# eda_emnist.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow_federated as tff
import collections

# ===============================
# LOAD EMNIST DATA
# ===============================
train_source, test_source = tff.simulation.datasets.emnist.load_data()
client_ids = train_source.client_ids
print(f"Total clients: {len(client_ids)}")

# Check first client data
example_dataset = train_source.create_tf_dataset_for_client(client_ids[0])
for example in example_dataset.take(1):
    print("Example keys:", list(example.keys()))
    print("Image shape:", example['pixels'].shape)
    print("Label:", example['label'].numpy())

# ===============================
# VISUALIZE SAMPLE IMAGES
# ===============================
def plot_samples(dataset, num_samples=12):
    plt.figure(figsize=(12,6))
    for i, example in enumerate(dataset.take(num_samples)):
        plt.subplot(3, 4, i+1)
        image = example['pixels'].numpy().reshape(28,28)
        label = example['label'].numpy()
        plt.imshow(image, cmap='gray')
        plt.title(f"Label: {label}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Plot sample images from first client
plot_samples(example_dataset)

# ===============================
# CLIENT-WISE CLASS DISTRIBUTION
# ===============================
def get_client_distribution(client_id):
    dataset = train_source.create_tf_dataset_for_client(client_id)
    labels = [example['label'].numpy() for example in dataset]
    return collections.Counter(labels)

# Example for first 5 clients
for i in range(5):
    dist = get_client_distribution(client_ids[i])
    print(f"Client {client_ids[i]} distribution: {dict(dist)}")

# ===============================
# GLOBAL CLASS DISTRIBUTION (TRAIN)
# ===============================
all_labels = []
for client_id in client_ids[:50]:  # first 50 clients for speed
    dataset = train_source.create_tf_dataset_for_client(client_id)
    all_labels.extend([example['label'].numpy() for example in dataset])

plt.figure(figsize=(10,5))
sns.countplot(x=all_labels)
plt.title("Global Class Distribution (Train - first 50 clients)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

# ===============================
# IMAGE STATISTICS
# ===============================
# Mean and std pixel values for first client
images = np.array([example['pixels'].numpy() for example in example_dataset])
print(f"Pixel value stats for first client:")
print(f"Mean: {images.mean():.4f}, Std: {images.std():.4f}, Min: {images.min()}, Max: {images.max()}")

# ===============================
# DONE
# ===============================
print("\n✅ EDA complete!")
