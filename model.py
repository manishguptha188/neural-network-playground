
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

SEED = 42
EPOCHS = 300

np.random.seed(SEED)
tf.random.set_seed(SEED)

# Generate the same synthetic dataset used by the playground.
X, y = make_moons(
    n_samples=500,
    noise=0.20,
    random_state=SEED
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=SEED,
    stratify=y
)

# Build our network: 2 inputs -> 8 ReLU neurons -> 1 sigmoid output.
model = keras.Sequential([
    keras.Input(shape=(2,)),
    layers.Dense(8, activation="relu", name="hidden_layer"),
    layers.Dense(1, activation="sigmoid", name="output_layer")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


class WeightHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.snapshots = {
            "hidden_weights": [],
            "hidden_biases": [],
            "output_weights": [],
            "output_biases": [],
        }

    def on_epoch_end(self, epoch, logs=None):
        hidden_w, hidden_b = (
            self.model.get_layer("hidden_layer").get_weights()
        )
        output_w, output_b = (
            self.model.get_layer("output_layer").get_weights()
        )

        self.snapshots["hidden_weights"].append(hidden_w.copy())
        self.snapshots["hidden_biases"].append(hidden_b.copy())
        self.snapshots["output_weights"].append(output_w.copy())
        self.snapshots["output_biases"].append(output_b.copy())


recorder = WeightHistory()

print(f"Training for {EPOCHS} epochs...")

history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=32,
    validation_split=0.20,
    callbacks=[recorder],
    verbose=0
)

# Evaluate only on the held-out test set.
test_probabilities = model.predict(X_test, verbose=0).ravel()
test_predictions = (test_probabilities >= 0.5).astype(int)

test_accuracy = accuracy_score(y_test, test_predictions)
test_loss, _ = model.evaluate(X_test, y_test, verbose=0)

print(f"\nTest accuracy: {test_accuracy:.2%}")
print(f"Test loss: {test_loss:.4f}")
print(f"Final training loss: {history.history['loss'][-1]:.4f}")
print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")

# Save the final trained model.
model.save("neural_network.keras")

# Save every epoch's parameters and training metrics.
np.savez_compressed(
    "training_snapshots.npz",
    hidden_weights=np.stack(recorder.snapshots["hidden_weights"]),
    hidden_biases=np.stack(recorder.snapshots["hidden_biases"]),
    output_weights=np.stack(recorder.snapshots["output_weights"]),
    output_biases=np.stack(recorder.snapshots["output_biases"]),
    training_loss=np.array(history.history["loss"]),
    validation_loss=np.array(history.history["val_loss"]),
    training_accuracy=np.array(history.history["accuracy"]),
    validation_accuracy=np.array(history.history["val_accuracy"]),
    test_accuracy=np.array([test_accuracy]),
    test_loss=np.array([test_loss]),
    test_predictions=test_predictions,
)

print("\nSaved neural_network.keras")
print(f"Saved {EPOCHS} epochs to training_snapshots.npz")
print("Done! Run: streamlit run app.py")
