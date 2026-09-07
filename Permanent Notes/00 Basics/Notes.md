# 00 · Basics

A plain list of what was covered before linear regression

---

## Learning from data

Normal software encodes rules a developer already knows. Machine learning is used
when the rules can't be written down (weather, fraud): collect many examples and
let the model fit them. The model is a set of tunable parameters ("knobs");
training adjusts them until predictions match reality.

## Problem types

- **Supervised**: learn from labelled examples (input paired with correct output).
  - **Regression**: predict a number. Example: house price.
  - **Classification**: predict which class.
    - **Binary**: two classes. Example: spam / not spam.
    - **Multiclass**: one of many classes. Example: handwritten digit 0 to 9.
    - **Hierarchical**: classes arranged in a tree. Example: animal, then dog, then breed.
  - **Tagging**: assign several labels to one input at once. Example: topics on an article.
  - **Search**: rank results by relevance to a query.
  - **Recommender systems**: rank items using personal preference. Example: film suggestions.
  - **Sequence learning**: input/output is a stream, order matters.
    - Parsing / tagging (input and output aligned).
    - Speech recognition (input and output not aligned).
    - Text to speech.
    - Translation.
- **Unsupervised / self-supervised**: no labels.
  - **Clustering**: group similar items. Example: customer segments.
  - **Dimensionality reduction (PCA)**: fewer parameters while keeping structure.
  - **Similarity in high-dimensional space**: measure closeness (Euclidean distance).
  - **Causal / probabilistic models**: cause-and-effect relationships.
  - **Generative models**: produce new samples like the training data.
- **Reinforcement learning**: take actions, receive rewards, adjust.
  - **Markov decision process**: the general framework (state, action, reward).
  - **Contextual bandit**: choose an option given some context.
  - **Multi-armed bandit**: no context; balance explore (try unknown) against exploit (use best known).

## The four components of a learning system

- **Data**: what the model is trained, validated and tested on.
- **Model**: the function mapping input to a prediction.
- **Objective (loss) function**: one number for how wrong a prediction is.
- **Optimization algorithm**: the rule that updates parameters.

## Model building blocks

- **Neuron**: multiplies each input by a weight, sums them, adds a bias, applies an
  activation function.
- **Weight**: how much an input counts toward the result.
- **Bias**: a per-neuron constant, independent of the inputs.
- **Activation function**: a non-linear squashing function. Example: sigmoid maps
  any number into (0, 1).
- **Layer**: a group of neurons reading the same inputs.
- **Network**: layers stacked so each feeds the next.

## Loss

- **Squared error**: (prediction minus target), squared, then averaged over the
  data. Squaring removes sign and penalises large misses more.

## Training

- **Gradient**: the direction in which loss increases fastest.
- **Gradient descent**: repeatedly step parameters opposite the gradient.
- **Learning rate**: the step size. Too small: slow. Too large: overshoots.
- **Backpropagation**: the chain rule applied layer by layer to get each
  parameter's gradient.
- **Epoch**: one pass over all the training data.
- **Automatic differentiation (autograd)**: the framework records operations as
  they run and replays them backward to compute gradients automatically.

## Batching

- **Full-batch**: all data per update. Accurate, expensive. Linear models with no
  activation also have a closed-form **analytic solution**,
  $w^{*} = (X^{\top} X)^{-1} X^{\top} y$.
- **Stochastic**: one example per update. Cheap, noisy, outlier-sensitive.
- **Mini-batch**: a small group per update. Averages out noise, fits in memory.
  The default choice.

## Linear algebra containers

- **Scalar**: a single number.
- **Vector**: a 1-D list of numbers.
- **Matrix**: a 2-D grid.
- **Tensor**: n-dimensional; the container that holds all of the above.
- **Matrix multiplication**: `B = A @ X`; read `A` as a transformation acting on
  `X`. Order matters; grouping does not (`(A @ X) @ Y = A @ (X @ Y)`). One neuron's
  weighted sum is a dot product; a layer is a matrix times a vector; a batch is a
  matrix times a matrix.

## Tools

- **NumPy**: `ndarray`, runs on CPU.
- **PyTorch / TensorFlow**: tensors, run on GPU (parallel, faster for large data).

## Probability

- **Bayes' theorem**: update belief after new evidence,
  $P(A \mid B) = P(B \mid A)\,P(A) / P(B)$.

---

## To revisit

- The rest of the statistics notation (deferred until a concept needs it).
- PCA and eigenvectors: the connection.
- Derive the analytic solution myself (set the loss derivative to zero).
- Reinforcement learning: Markov decision process details.
