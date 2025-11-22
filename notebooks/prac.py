
# ===============================
# LOAD EMNIST DATA
# ===============================
source, test_source = tff.simulation.datasets.emnist.load_data()

def client_data(n):
    return source.create_tf_dataset_for_client(source.client_ids[n]).map(
        lambda e: (tf.reshape(e['pixels'], [-1]), e['label'])
    ).repeat(EPOCHS_PER_CLIENT).batch(BATCH_SIZE)

train_data = [client_data(n) for n in range(NUM_CLIENTS)]

def preprocess_test(example):
    return tf.reshape(example['pixels'], [-1]), example['label']

test_ds = test_source.create_tf_dataset_from_all_clients().map(preprocess_test).batch(BATCH_SIZE)

# ===============================
# DEFINE MODEL
# ===============================
keras_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)), 
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

tff_model = tff.learning.models.functional_model_from_keras(
    keras_model,
    loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(),
    input_spec=train_data[0].element_spec,
    metrics_constructor=collections.OrderedDict(
        accuracy=tf.keras.metrics.SparseCategoricalAccuracy)
)


# ===============================
# FEDERATED TRAINING
# ===============================
trainer = tff.learning.algorithms.build_weighted_fed_avg(
    tff_model,
    client_optimizer_fn=tff.learning.optimizers.build_sgdm(learning_rate=LEARNING_RATE)
)

state = trainer.initialize()
accuracies = []
test_accuracies = []  # New list to store test accuracies over rounds
import time
import pandas as pd
# Track per-client accuracies over rounds
client_accuracies_over_rounds = {i: [] for i in range(NUM_CLIENTS)}
round_durations = []
drift_scores = []  # Simple drift: change in global test accuracy

print("Starting Federated Training...\n")
for round_num in tqdm(range(1, NUM_ROUNDS + 1), desc="Rounds"):
    start_time = time.time()
    result = trainer.next(state, train_data)
    state = result.state
    metrics = result.metrics
    accuracy = metrics['client_work']['train']['accuracy']
    accuracies.append(accuracy)

    # Evaluate test accuracy on the current global model after each round
    temp_keras_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    model_weights = state.global_model_weights
    for var, val in zip(temp_keras_model.trainable_variables, model_weights.trainable):
        var.assign(val)
    for var, val in zip(temp_keras_model.non_trainable_variables, model_weights.non_trainable):
        var.assign(val)
    temp_keras_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    _, test_acc = temp_keras_model.evaluate(test_ds, verbose=0)
    test_accuracies.append(test_acc)

    print(f"\nRound {round_num} training accuracy: {accuracy:.4f}, test accuracy: {test_acc:.4f}")
    if round_num > 1 and accuracy < ACCURACY_THRESHOLD:
        print("WARNING: Potential global data drift detected!")
    else:
        print("No global drift detected.")
    for client_id in range(NUM_CLIENTS):
        client_accuracies_over_rounds[client_id].append(accuracy) 
    round_duration = time.time() - start_time
    round_durations.append(round_duration)
    if round_num > 1:
        drift = test_acc - test_accuracies[-2]
        drift_scores.append(drift)
    else:
        drift_scores.append(0)


pd.DataFrame({
      "Round": list(range(1, NUM_ROUNDS + 1)),
      "Training_Accuracy": accuracies,  # Add underscores
      "Test_Accuracy": test_accuracies,
      "Round_Duration": round_durations,  # Simplified name
      "Drift_Score": drift_scores
  }).to_csv("outputs/global_metrics.csv", index=False)
pd.DataFrame(client_accuracies_over_rounds).to_csv('outputs/client_accuracies.csv',index=False)


# ===============================
# EXTRACT TRAINED GLOBAL MODEL
# ===============================
keras_model_trained = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

model_weights = state.global_model_weights
for var, val in zip(keras_model_trained.trainable_variables, model_weights.trainable):
    var.assign(val)
for var, val in zip(keras_model_trained.non_trainable_variables, model_weights.non_trainable):
    var.assign(val)

keras_model_trained.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)

# ===============================
# CENTRALIZED EVALUATION
# ===============================
print("\nEvaluating on centralized test set...")
test_loss, test_accuracy = keras_model_trained.evaluate(test_ds, verbose=0)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# ===============================
# CLASSIFICATION REPORT & CONFUSION MATRIX
# ===============================
all_preds = []
all_true = []

for x, y in test_ds:
    preds = keras_model_trained.predict(x, verbose=0)
    all_preds.extend(np.argmax(preds, axis=1))
    all_true.extend(y.numpy())

# Save classification report
report = classification_report(all_true, all_preds)
with open("outputs/classification_report.txt", "w") as f:
    f.write(report)

# Confusion matrix
cm = confusion_matrix(all_true, all_preds)

plt.figure(figsize=(12,12))
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix")
plt.colorbar()
plt.savefig("outputs/confusion_matrix.png")
plt.close()

print("Saved classification report & confusion matrix.")

# ===============================
# CLASS DISTRIBUTION PLOT (GLOBAL TEST DATA)
# ===============================
plt.figure(figsize=(12,6))
plt.hist(all_true, bins=62)
plt.title("Global Test Set Class Distribution")
plt.savefig("outputs/global_class_distribution.png")
plt.close()

# ===============================
# PER-USER PERSONALIZATION & METRICS
# ===============================
print("\n--- Per-User Personalization & Dashboard Metrics ---")
user_metrics = {}

for user_id in range(NUM_CLIENTS):
    print(f"\nUser {user_id}:")

    user_ds_raw = source.create_tf_dataset_for_client(source.client_ids[user_id])
    user_ds_for_eval = user_ds_raw.map(
        lambda e: (tf.reshape(e['pixels'], [-1]), e['label'])
    ).batch(BATCH_SIZE)

    user_loss, user_accuracy = keras_model_trained.evaluate(user_ds_for_eval, verbose=0)
    print(f"  Accuracy on own data: {user_accuracy:.4f}")

    # Save per-user class distribution
    user_labels = [x['label'].numpy() for x in user_ds_raw]
    plt.figure(figsize=(12,6))
    plt.hist(user_labels, bins=62)
    plt.title(f"User {user_id} Class Distribution")
    plt.savefig(f"outputs/user_{user_id}_class_distribution.png")
    plt.close()

    # Personalization
    personalized_model = tf.keras.models.clone_model(keras_model_trained)
    personalized_model.set_weights(keras_model_trained.get_weights())
    personalized_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE / 10),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

    user_ds = user_ds_for_eval.unbatch().batch(1000)
    for batch in user_ds.take(1):
        x_batch, y_batch = batch
        personalized_model.fit(x_batch, y_batch, epochs=1, verbose=0)

    personalized_acc = personalized_model.evaluate(user_ds, verbose=0)[1]
    print(f"  Personalized Accuracy: {personalized_acc:.4f}")

    user_metrics[user_id] = {
        "accuracy": user_accuracy,
        "personalized_accuracy": personalized_acc,
    }

    # Save personalized model
    personalized_model.save(f"models/personalized_user_{user_id}.h5")

# ===============================
# PLOT ACCURACY OVER ROUNDS (UPDATED: Now includes test accuracy)
# ===============================
plt.figure(figsize=(10, 6))
plt.plot(range(1, NUM_ROUNDS + 1), accuracies, marker='o', label='Training Accuracy')
plt.plot(range(1, NUM_ROUNDS + 1), test_accuracies, marker='s', label='Test Accuracy')
plt.title('Global Model: Training and Test Accuracy Over Rounds')
plt.xlabel('Round')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('outputs/global_accuracy_over_rounds.png')
plt.close()

print("Saved global accuracy over rounds plot.")

# ===============================
# PLOT PER-CLIENT ACCURACIES
# ===============================
client_ids = list(user_metrics.keys())
global_accs = [user_metrics[id]['accuracy'] for id in client_ids]
personalized_accs = [user_metrics[id]['personalized_accuracy'] for id in client_ids]

plt.figure(figsize=(10, 6))
x = np.arange(len(client_ids))
plt.bar(x - 0.2, global_accs, 0.4, label='Global Model Accuracy')
plt.bar(x + 0.2, personalized_accs, 0.4, label='Personalized Accuracy')
plt.xticks(x, [f'Client {i}' for i in client_ids])
plt.title('Per-Client Accuracies: Global vs. Personalized')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('outputs/per_client_accuracies.png')
plt.close()

print("Saved per-client accuracies plot.")

# ===============================
# SAVE GLOBAL MODEL
# ===============================
keras_model_trained.save("models/emnist_federated_model.h5")
print("\nAll models saved in /models and plots in /outputs.")
