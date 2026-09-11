"""
forward pass, MSE, gradients, SGD update, regularization
"""

import numpy as np
import matplotlib.pyplot as plt


class SyntheticData:
    """Normally distributed data for  y = x1*w1 + x2*w2 + b + noise."""

    def __init__(self, n=200, test_ratio=0.3, batch_size=16,
                 true_w=(2.0, -3.4), true_b=4.2, noise_std=1.0, seed=1):
        rng = np.random.default_rng(seed)
        self.true_w = np.array(true_w)
        self.true_b = true_b
        self.batch_size = batch_size
        self._rng = rng

        # std = spread: ~68% of values within mean +/-1 std, ~95% within +/-2, ~99.7% within +/-3
        X = rng.normal(0.0, 1.0, size=(n, 2))
        noise = rng.normal(0.0, noise_std, size=n)
        y = X @ self.true_w + true_b + noise

        split = int(n * (1 - test_ratio))
        self.X_train, self.y_train = X[:split], y[:split]
        self.X_test, self.y_test = X[split:], y[split:]

    def train_batches(self):
        """Yield shuffled mini-batches for one epoch."""
        order = self._rng.permutation(len(self.X_train))
        for start in range(0, len(order), self.batch_size):
            idx = order[start:start + self.batch_size]
            yield self.X_train[idx], self.y_train[idx]


class LinearRegression:
    """Two weights, one bias. Explicit forward and backward math."""

    def __init__(self, weight_decay=0.0):
        self.w = np.zeros(2)
        self.b = 0.0
        self.weight_decay = weight_decay  # L2 lambda; 0 disables it

    def predict(self, X):
        return X @ self.w + self.b

    def mse(self, X, y):
        return np.mean((self.predict(X) - y) ** 2)

    def loss(self, X, y):
        """Training objective: MSE plus the L2 penalty on weights."""
        l2_penalty = self.weight_decay * np.sum(self.w ** 2) / 2
        return self.mse(X, y) + l2_penalty

    def gradients(self, X, y):
        """d(loss)/d(w), d(loss)/d(b) for MSE + L2. Bias is not decayed."""
        error = self.predict(X) - y
        n = len(y)
        grad_w = (2 / n) * (X.T @ error) + self.weight_decay * self.w
        grad_b = (2 / n) * np.sum(error)
        return grad_w, grad_b


class Trainer:
    """Runs mini-batch SGD and records train/test loss per epoch."""

    def __init__(self, model, data, lr=0.05, epochs=40):
        self.model = model
        self.data = data
        self.lr = lr
        self.epochs = epochs

    def fit(self, verbose=True):
        history = {"train": [], "test": []}

        for epoch in range(1, self.epochs + 1):
            for X, y in self.data.train_batches():
                grad_w, grad_b = self.model.gradients(X, y)
                self.model.w -= self.lr * grad_w
                self.model.b -= self.lr * grad_b

            train_mse = self.model.mse(self.data.X_train, self.data.y_train)
            test_mse = self.model.mse(self.data.X_test, self.data.y_test)
            history["train"].append(train_mse)
            history["test"].append(test_mse)

            if verbose:
                print(f"epoch {epoch:02d} | train mse {train_mse:.4f} | test mse {test_mse:.4f}")

        return history


def plot_runs(runs, data):
    """runs: {label: (model, history)}."""
    fig, (ax_loss, ax_norm) = plt.subplots(1, 2, figsize=(11, 4))

    for label, (_, history) in runs.items():
        line, = ax_loss.plot(history["train"], label=f"{label} train")
        ax_loss.plot(history["test"], "--", color=line.get_color(),
                     label=f"{label} test")
    ax_loss.set(title="Loss over epochs", xlabel="epoch", ylabel="MSE")
    ax_loss.legend()

    labels = list(runs)
    norms = [float(np.sum(runs[l][0].w ** 2)) for l in labels]
    ax_norm.bar(labels, norms, color=["tab:blue", "tab:orange"])
    ax_norm.set(title="Weight size after training", ylabel="||w||^2")

    plt.tight_layout()
    plt.show()


def main():
    # Few noisy samples so overfitting (and the effect of decay) is visible.
    data = SyntheticData(n=40, noise_std=2.0)

    runs = {}
    for label, wd in {"no decay": 0.0, "L2 wd=0.1": 0.1}.items():
        print(f"\n--- {label} ---")
        model = LinearRegression(weight_decay=wd)
        history = Trainer(model, data).fit()
        runs[label] = (model, history)
        print(f"w = {model.w.round(3)} (true {data.true_w})  "
              f"b = {model.b:.3f} (true {data.true_b})  "
              f"||w||^2 = {np.sum(model.w ** 2):.3f}")

    plot_runs(runs, data)


if __name__ == "__main__":
    main()
