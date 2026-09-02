"""
The PyTorch layer.

Same task as From_Scratch.py. PyTorch now handles gradients (loss.backward())
and gives us pre-built pieces: nn.Linear (model), nn.MSELoss (loss),
optim.SGD (update). We still write the training loop.

--- one level deeper (what nn.Linear / optim.SGD wrap) -----------------------
Without those pieces you would manage raw parameter tensors and step them by
hand. The core of it:

    w = torch.zeros(2, requires_grad=True)     # instead of nn.Linear
    b = torch.zeros(1, requires_grad=True)
    pred = X @ w + b                           # instead of model(X)
    loss = ((pred - y) ** 2).mean() + wd / 2 * (w ** 2).sum()   # + L2 by hand

    loss.backward()                            # this part is the same
    with torch.no_grad():                      # instead of optimizer.step()
        for p in (w, b):
            p -= lr * p.grad
            p.grad.zero_()                     # instead of optimizer.zero_grad()

So optim.SGD(weight_decay=...) is just  "p -= lr * (p.grad + wd * p)"  looped
over every parameter. nn.Linear is just the two tensors plus  X @ w + b.
---------------------------------------------------------------------------
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


class SyntheticData:
    """Normally distributed data for  y = x1*w1 + x2*w2 + b + noise."""

    def __init__(self, n=200, test_ratio=0.3, batch_size=16,
                 true_w=(2.0, -3.4), true_b=4.2, noise_std=1.0, seed=1):
        gen = torch.Generator().manual_seed(seed)
        self.true_w = torch.tensor(true_w)
        self.true_b = true_b
        self.batch_size = batch_size
        self._gen = gen

        X = torch.normal(0.0, 1.0, (n, 2), generator=gen)
        noise = torch.normal(0.0, noise_std, (n,), generator=gen)
        y = X @ self.true_w + true_b + noise

        split = int(n * (1 - test_ratio))
        self.train_ds = TensorDataset(X[:split], y[:split])
        self.test_ds = TensorDataset(X[split:], y[split:])

    def train_loader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                          shuffle=True, generator=self._gen)

    def full(self, train=True):
        return (self.train_ds if train else self.test_ds).tensors


class LinearRegression(nn.Module):
    """Two inputs -> one output; bias is included by default."""

    def __init__(self):
        super().__init__()
        self.net = nn.Linear(2, 1)  # deeper level: a (2,) weight + (1,) bias tensor

    def forward(self, X):
        return self.net(X).squeeze(-1)  # deeper level: X @ w + b


class Trainer:
    """Mini-batch SGD via autograd; records train/test MSE per epoch."""

    def __init__(self, model, data, lr=0.05, epochs=40, weight_decay=0.0):
        self.model = model
        self.data = data
        self.epochs = epochs
        self.loss_fn = nn.MSELoss()
        # L2 on weights only, not bias (matches From_Scratch.py)
        self.optimizer = torch.optim.SGD([
            {"params": model.net.weight, "weight_decay": weight_decay},
            {"params": model.net.bias, "weight_decay": 0.0},
        ], lr=lr)

    @torch.no_grad()
    def mse(self, train=True):
        X, y = self.data.full(train)
        return self.loss_fn(self.model(X), y).item()

    def fit(self, verbose=True):
        history = {"train": [], "test": []}

        for epoch in range(1, self.epochs + 1):
            for X, y in self.data.train_loader():
                self.optimizer.zero_grad()               # deeper: p.grad.zero_()
                loss = self.loss_fn(self.model(X), y)
                loss.backward()                          # same at every level
                self.optimizer.step()                    # deeper: p -= lr * p.grad

            history["train"].append(self.mse(train=True))
            history["test"].append(self.mse(train=False))

            if verbose:
                print(f"epoch {epoch:02d} | train mse {history['train'][-1]:.4f} "
                      f"| test mse {history['test'][-1]:.4f}")

        return history


def weight_norm_sq(model):
    return (model.net.weight.detach() ** 2).sum().item()


def plot_runs(runs):
    """runs: {label: (model, history)}."""
    fig, (ax_loss, ax_norm) = plt.subplots(1, 2, figsize=(11, 4))

    for label, (_, history) in runs.items():
        line, = ax_loss.plot(history["train"], label=f"{label} train")
        ax_loss.plot(history["test"], "--", color=line.get_color(),
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
    torch.manual_seed(0)
    data = SyntheticData(n=40, noise_std=2.0)

    runs = {}
    for label, wd in {"no decay": 0.0, "L2 wd=0.1": 0.1}.items():
        print(f"\n--- {label} ---")
        model = LinearRegression()
        history = Trainer(model, data, weight_decay=wd).fit()
        runs[label] = (model, history)

        w = model.net.weight.detach().flatten()
        b = model.net.bias.item()
        print(f"w = {w.numpy().round(3)} (true {data.true_w.numpy()})  "
              f"b = {b:.3f} (true {data.true_b})  "
              f"||w||^2 = {weight_norm_sq(model):.3f}")

    plot_runs(runs)


if __name__ == "__main__":
    main()
