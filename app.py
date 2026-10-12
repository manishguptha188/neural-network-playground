import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------
st.set_page_config(
    page_title="Neural Network Playground",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧠 Neural Network Playground")
st.caption("Explore how a small neural network learns to separate two classes, epoch by epoch.")

SEED = 42
TOTAL_EPOCHS_FALLBACK = 300
SNAPSHOT_PATH = "training_snapshots.npz"


# ------------------------------------------------------------
# Load saved training snapshots
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_snapshots(path: str):
    """Load NPZ contents into a normal dictionary of NumPy arrays."""
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def find_key(data, aliases):
    """Return the first matching key from a set of possible key names."""
    for name in aliases:
        if name in data:
            return name
    return None


def require_array(data, aliases, friendly_name):
    key = find_key(data, aliases)
    if key is None:
        st.error(
            f"The saved file is missing **{friendly_name}**. "
            "Please run `python model.py` once to regenerate `training_snapshots.npz`, "
            "then restart this app."
        )
        st.caption("Keys found in the file: " + (", ".join(data.keys()) if data else "none"))
        st.stop()
    return np.asarray(data[key])


try:
    snapshots = load_snapshots(SNAPSHOT_PATH)
except FileNotFoundError:
    st.error(f"Could not find `{SNAPSHOT_PATH}` in this project folder.")
    st.info("Open the project terminal and run `python model.py` first.")
    st.stop()
except Exception as exc:
    st.error(f"Could not read `{SNAPSHOT_PATH}`: {exc}")
    st.stop()


# ------------------------------------------------------------
# Read history arrays. Support both current and common older names.
# ------------------------------------------------------------
training_accuracy = require_array(
    snapshots,
    ["training_accuracy", "train_accuracy", "accuracy"],
    "training accuracy history",
).reshape(-1)
validation_accuracy = require_array(
    snapshots,
    ["validation_accuracy", "val_accuracy", "validation_acc"],
    "validation accuracy history",
).reshape(-1)
training_loss = require_array(
    snapshots,
    ["training_loss", "train_loss", "loss"],
    "training loss history",
).reshape(-1)
validation_loss = require_array(
    snapshots,
    ["validation_loss", "val_loss"],
    "validation loss history",
).reshape(-1)

hidden_weights = require_array(
    snapshots,
    ["hidden_weights", "hidden_layer_weights", "hidden_kernel", "hidden_layer_kernel", "W1", "w1"],
    "hidden-layer weights",
)
hidden_biases = require_array(
    snapshots,
    ["hidden_biases", "hidden_layer_biases", "hidden_bias", "hidden_layer_bias", "b1", "B1"],
    "hidden-layer biases",
)
output_weights = require_array(
    snapshots,
    ["output_weights", "output_layer_weights", "output_kernel", "output_layer_kernel", "W2", "w2"],
    "output-layer weights",
)
output_biases = require_array(
    snapshots,
    ["output_biases", "output_layer_biases", "output_bias", "output_layer_bias", "b2", "B2"],
    "output-layer biases",
)


def as_epoch_array(array, expected_epochs, friendly_name):
    """Normalize a saved weight history to an array indexed by epoch."""
    array = np.asarray(array)
    if array.ndim == 0:
        st.error(f"The saved {friendly_name} has no epoch dimension. Run `python model.py` again.")
        st.stop()
    if array.shape[0] != expected_epochs:
        st.error(
            f"The saved {friendly_name} contains {array.shape[0]} entries, but the training history "
            f"contains {expected_epochs} epochs. Run `python model.py` again so the files match."
        )
        st.stop()
    return array


total_epochs = min(
    len(training_accuracy),
    len(validation_accuracy),
    len(training_loss),
    len(validation_loss),
)
if total_epochs < 1:
    st.error("No training history was found. Run `python model.py` to train and save the network.")
    st.stop()

training_accuracy = training_accuracy[:total_epochs]
validation_accuracy = validation_accuracy[:total_epochs]
training_loss = training_loss[:total_epochs]
validation_loss = validation_loss[:total_epochs]
hidden_weights = as_epoch_array(hidden_weights, total_epochs, "hidden-layer weights")
hidden_biases = as_epoch_array(hidden_biases, total_epochs, "hidden-layer biases")
output_weights = as_epoch_array(output_weights, total_epochs, "output-layer weights")
output_biases = as_epoch_array(output_biases, total_epochs, "output-layer biases")


# ------------------------------------------------------------
# Recreate the same synthetic dataset and train/test split
# ------------------------------------------------------------
X, y = make_moons(n_samples=500, noise=0.20, random_state=SEED)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=SEED,
)


def create_model():
    model = keras.Sequential(
        [
            keras.Input(shape=(2,)),
            layers.Dense(8, activation="relu", name="hidden_layer"),
            layers.Dense(1, activation="sigmoid", name="output_layer"),
        ]
    )
    return model


def load_epoch_model(epoch_index):
    model = create_model()
    model.set_weights(
        [
            hidden_weights[epoch_index],
            hidden_biases[epoch_index],
            output_weights[epoch_index],
            output_biases[epoch_index],
        ]
    )
    return model


# ------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------
st.sidebar.header("Controls")
selected_epoch = st.sidebar.slider(
    "Training epoch",
    min_value=1,
    max_value=total_epochs,
    value=total_epochs,
    step=1,
    help="Move this slider to see how the decision boundary changes as training progresses.",
)
st.sidebar.caption(f"Showing the saved parameters after epoch {selected_epoch} of {total_epochs}.")


epoch_index = selected_epoch - 1
model = load_epoch_model(epoch_index)


# ------------------------------------------------------------
# Decision boundary and selected input
# ------------------------------------------------------------
x1_min, x1_max = X[:, 0].min() - 0.6, X[:, 0].max() + 0.6
x2_min, x2_max = X[:, 1].min() - 0.6, X[:, 1].max() + 0.6
grid_x1, grid_x2 = np.meshgrid(
    np.linspace(x1_min, x1_max, 180),
    np.linspace(x2_min, x2_max, 180),
)
grid_points = np.c_[grid_x1.ravel(), grid_x2.ravel()]
grid_probs = model.predict(grid_points, verbose=0).reshape(grid_x1.shape)

st.sidebar.divider()
st.sidebar.subheader("Try an input point")
feature_1 = st.sidebar.slider("Feature 1", float(x1_min), float(x1_max), 0.0, 0.05)
feature_2 = st.sidebar.slider("Feature 2", float(x2_min), float(x2_max), 0.0, 0.05)
sample = np.array([[feature_1, feature_2]], dtype=np.float32)
probability = float(model.predict(sample, verbose=0)[0, 0])
predicted_class = int(probability >= 0.5)


# ------------------------------------------------------------
# Summary metrics
# ------------------------------------------------------------
metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("Training epoch", f"{selected_epoch}/{total_epochs}")
metric2.metric("Training accuracy", f"{training_accuracy[epoch_index]:.1%}")
metric3.metric("Validation accuracy", f"{validation_accuracy[epoch_index]:.1%}")
metric4.metric("Predicted class", f"Class {predicted_class}")

st.write(
    f"For the point **({feature_1:.2f}, {feature_2:.2f})**, "
    f"the network assigns **{probability:.1%}** probability to class 1. "
    f"Its predicted class is **{predicted_class}**."
)


# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------
tab_network, tab_boundary, tab_activations, tab_history, tab_weights = st.tabs(
    ["🧠 Network", "🌗 Decision Boundary", "⚡ Neuron Activations", "📈 Training History", "🔢 Weights & Biases"]
)

with tab_network:
    st.subheader("The network at this epoch")
    st.markdown(
        "**2 input features → 8 ReLU neurons → 1 sigmoid output**\n\n"
        "Each connection carries a learned weight. The biases shift each neuron's activation. "
        "Use the epoch slider in the sidebar to inspect how the saved parameters evolve."
    )

    # A simple architecture diagram made with Plotly, without extra dependencies.
    fig_net = go.Figure()
    input_x, hidden_x, output_x = 0.08, 0.50, 0.92
    input_y = [0.62, 0.38]
    hidden_y = np.linspace(0.12, 0.88, 8).tolist()
    output_y = [0.50]

    for iy in input_y:
        for hy in hidden_y:
            fig_net.add_trace(go.Scatter(
                x=[input_x, hidden_x], y=[iy, hy], mode="lines",
                line=dict(color="rgba(120,130,150,0.28)", width=1),
                hoverinfo="skip", showlegend=False
            ))
    for hy in hidden_y:
        fig_net.add_trace(go.Scatter(
            x=[hidden_x, output_x], y=[hy, output_y[0]], mode="lines",
            line=dict(color="rgba(120,130,150,0.35)", width=1),
            hoverinfo="skip", showlegend=False
        ))

    fig_net.add_trace(go.Scatter(
        x=[input_x] * 2, y=input_y, mode="markers+text",
        marker=dict(size=24), text=["x₁", "x₂"], textposition="middle left",
        name="Inputs", hovertemplate="Input feature<extra></extra>"
    ))
    fig_net.add_trace(go.Scatter(
        x=[hidden_x] * 8, y=hidden_y, mode="markers+text",
        marker=dict(size=20), text=[f"h{i+1}" for i in range(8)], textposition="middle center",
        name="Hidden neurons", hovertemplate="Hidden neuron<extra></extra>"
    ))
    fig_net.add_trace(go.Scatter(
        x=[output_x], y=output_y, mode="markers+text",
        marker=dict(size=28), text=["ŷ"], textposition="middle left",
        name="Output", hovertemplate="Predicted probability<extra></extra>"
    ))
    fig_net.update_layout(
        height=420,
        margin=dict(l=25, r=25, t=20, b=20),
        xaxis=dict(visible=False, range=[-0.03, 1.03]),
        yaxis=dict(visible=False, range=[0, 1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig_net, use_container_width=True)

with tab_boundary:
    st.subheader("How the network separates the classes")
    fig_boundary = go.Figure()
    fig_boundary.add_trace(go.Contour(
        x=np.linspace(x1_min, x1_max, 180),
        y=np.linspace(x2_min, x2_max, 180),
        z=grid_probs,
        contours=dict(start=0, end=1, size=0.1, showlines=False),
        colorbar=dict(title="P(class 1)"),
        opacity=0.72,
        hovertemplate="Feature 1: %{x:.2f}<br>Feature 2: %{y:.2f}<br>P(class 1): %{z:.2f}<extra></extra>",
        name="Predicted probability",
    ))
    for class_value, label in [(0, "Class 0"), (1, "Class 1")]:
        mask = y == class_value
        fig_boundary.add_trace(go.Scatter(
            x=X[mask, 0], y=X[mask, 1], mode="markers", name=label,
            marker=dict(size=7, line=dict(width=0.5, color="white")),
            hovertemplate="Feature 1: %{x:.2f}<br>Feature 2: %{y:.2f}<extra></extra>",
        ))
    fig_boundary.add_trace(go.Scatter(
        x=[feature_1], y=[feature_2], mode="markers+text", name="Your input",
        text=[f"Predicted: {predicted_class}"], textposition="top center",
        marker=dict(size=16, symbol="star", line=dict(width=1.5, color="white")),
    ))
    fig_boundary.update_layout(
        height=600,
        xaxis_title="Feature 1",
        yaxis_title="Feature 2",
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig_boundary, use_container_width=True)

with tab_activations:
    st.subheader("What the hidden neurons respond to")
    hidden_layer_model = keras.Model(inputs=model.inputs, outputs=model.get_layer("hidden_layer").output)
    activations = hidden_layer_model.predict(sample, verbose=0)[0]
    activation_df = pd.DataFrame({
        "Neuron": [f"Neuron {i+1}" for i in range(len(activations))],
        "Activation": activations,
    })
    fig_activations = go.Figure(go.Bar(
        x=activation_df["Neuron"], y=activation_df["Activation"],
        hovertemplate="%{x}<br>Activation: %{y:.4f}<extra></extra>",
    ))
    fig_activations.update_layout(
        height=400, xaxis_title="Hidden neuron", yaxis_title="ReLU activation",
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig_activations, use_container_width=True)
    st.caption("ReLU neurons output zero for negative pre-activation values and positive values otherwise.")

with tab_history:
    st.subheader("Learning over time")
    epochs_axis = np.arange(1, total_epochs + 1)
    col_loss, col_acc = st.columns(2)
    with col_loss:
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(x=epochs_axis, y=training_loss, mode="lines", name="Training loss"))
        fig_loss.add_trace(go.Scatter(x=epochs_axis, y=validation_loss, mode="lines", name="Validation loss"))
        fig_loss.add_vline(x=selected_epoch, line_dash="dash", annotation_text=f"Epoch {selected_epoch}")
        fig_loss.update_layout(height=380, title="Loss", xaxis_title="Epoch", yaxis_title="Binary cross-entropy")
        st.plotly_chart(fig_loss, use_container_width=True)
    with col_acc:
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(x=epochs_axis, y=training_accuracy, mode="lines", name="Training accuracy"))
        fig_acc.add_trace(go.Scatter(x=epochs_axis, y=validation_accuracy, mode="lines", name="Validation accuracy"))
        fig_acc.add_vline(x=selected_epoch, line_dash="dash", annotation_text=f"Epoch {selected_epoch}")
        fig_acc.update_layout(height=380, title="Accuracy", xaxis_title="Epoch", yaxis_title="Accuracy", yaxis_tickformat=".0%")
        st.plotly_chart(fig_acc, use_container_width=True)

with tab_weights:
    st.subheader(f"Parameters saved after epoch {selected_epoch}")
    current_hidden_weights = hidden_weights[epoch_index]
    current_hidden_biases = hidden_biases[epoch_index]
    current_output_weights = output_weights[epoch_index]
    current_output_biases = output_biases[epoch_index]

    left, right = st.columns(2)
    with left:
        st.markdown("**Input → hidden layer weights**")
        st.dataframe(pd.DataFrame(
            current_hidden_weights,
            index=["Feature 1", "Feature 2"],
            columns=[f"Neuron {i+1}" for i in range(current_hidden_weights.shape[1])],
        ).round(4), use_container_width=True)
        st.markdown("**Hidden layer biases**")
        st.dataframe(pd.DataFrame([current_hidden_biases], columns=[f"Neuron {i+1}" for i in range(len(current_hidden_biases))]).round(4), use_container_width=True)
    with right:
        st.markdown("**Hidden → output layer weights**")
        st.dataframe(pd.DataFrame(
            current_output_weights,
            index=[f"Neuron {i+1}" for i in range(current_output_weights.shape[0])],
            columns=["Output"],
        ).round(4), use_container_width=True)
        st.markdown("**Output bias**")
        st.dataframe(pd.DataFrame([current_output_biases.reshape(-1)], columns=["Output bias"]).round(4), use_container_width=True)


# ------------------------------------------------------------
# Final saved evaluation metric (when present)
# ------------------------------------------------------------
st.divider()
if "test_accuracy" in snapshots:
    saved_test_accuracy = float(np.asarray(snapshots["test_accuracy"]).reshape(-1)[0])
    st.caption(f"Final saved model's held-out test accuracy: {saved_test_accuracy:.1%}. This is the final model's score, not necessarily the selected epoch's score.")
elif "val_accuracy" in snapshots or "validation_accuracy" in snapshots:
    st.caption("Held-out test accuracy was not saved in this snapshot file. The validation accuracy above is shown for the selected epoch.")
else:
    st.caption("Test accuracy is unavailable in the current snapshot file.")

st.caption("Educational demo using a synthetic two-class dataset; results are for learning, not a production classifier.")
