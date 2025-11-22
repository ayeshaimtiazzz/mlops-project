from keras.models import load_model
from keras.losses import SparseCategoricalCrossentropy

model_path = r"C:\\Users\\aimte\\OneDrive\\Desktop\\uni work\\mlops\\project_mlops\\notebooks\\models\\emnist_federated_model.h5"
model = load_model(
    model_path,
    custom_objects={"SparseCategoricalCrossentropy": SparseCategoricalCrossentropy}
)

print(f"Model loaded successfully from {model_path}")