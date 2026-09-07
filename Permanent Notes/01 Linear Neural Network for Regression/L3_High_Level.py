"""
The high-level layer: the framework runs the loop too.

Same task again, but there is no training loop in this file. We describe the
model, compile it with an optimizer + loss, and call model.fit(). Keras owns
the epoch/batch/backward/step cycle.

(PyTorch-ecosystem equivalents: PyTorch Lightning, skorch, fastai.)
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import keras
import matplotlib.pyplot as plt


class SyntheticData:
    """Normally distributed data for  y = x1*w1 + x2*w2 + b + noise."""

    def __init__(self, n=200, test_ratio=0.3, batch_size=16,
                 true_w=(2.0, -3.4), true_b=4.2, noise_std=1.0, seed=1):
        rng = np.random.default_rng(seed)
        self.true_w = np.array(true_w)
        self.true_b = true_b
        self.batch_size = batch_size

        X = rng.normal(0.0, 1.0, size=(n, 2))
        noise = rng.normal(0.0, noise_std, size=n)
        y = X @ self.true_w + true_b + noise

        split = int(n * (1 - test_ratio))
        self.X_train, self.y_train = X[:split], y[:split]
        self.X_test, self.y_test = X[split:], y[split:]


def build_model(weight_decay=0.0, lr=0.05):
    """Two inputs -> one output. L2 on the kernel (weights), not the bias."""
    model = keras.Sequential([
        keras.layers.Input((2,)),
        keras.layers.Dense(1, kernel_regularizer=keras.regularizers.l2(weight_decay / 2)),
    ])
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=lr),
                  loss="mse", metrics=["mse"])
    return model


def weight_norm_sq(model):
    w = model.layers[0].kernel.numpy()
    return float((w ** 2).sum())


def plot_runs(runs):
    """runs: {label: (model, history)}."""
    fig, (ax_loss, ax_norm) = plt.subplots(1, 2, figsize=(11, 4))

    for label, (_, history) in runs.items():
        line, = ax_loss.plot(history["mse"], label=f"{label} train")
        ax_loss.plot(history["val_mse"], "--", color=line.get_color(),
                     label=f"{label} test")
    ax_loss.set(title="Loss over epochs", xlabel="epoch", ylabel="MSE")
    ax_loss.legend()

    labels = list(runs)
    norms = [weight_norm_sq(runs[l][0]) for l in labels]
    ax_norm.bar(labels, norms, color=["tab:blue", "tab:orange"])
    ax_norm.set(title="Weight size after training", ylabel="||w||^2")

    plt.tight_layout()
    plt.show()


def main():
    keras.utils.set_random_seed(0)
    data = SyntheticData(n=40, noise_std=2.0)

    runs = {}
    for label, wd in {"no decay": 0.0, "L2 wd=0.1": 0.1}.items():
        print(f"\n--- {label} ---")
        model = build_model(weight_decay=wd)
        hist = model.fit(
            data.X_train, data.y_train,
            validation_data=(data.X_test, data.y_test),
            batch_size=data.batch_size, epochs=40, verbose=0,
        )
        runs[label] = (model, hist.history)

        w = model.layers[0].kernel.numpy().flatten().round(3)
        b = float(model.layers[0].bias.numpy()[0])
        print(f"w = {w} (true {data.true_w})  "
              f"b = {b:.3f} (true {data.true_b})  "
              f"||w||^2 = {weight_norm_sq(model):.3f}")

    plot_runs(runs)


if __name__ == "__main__":
    main()
