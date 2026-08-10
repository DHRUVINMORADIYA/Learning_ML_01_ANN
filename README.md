### Why do we need Artificial Neural Networks?
    
In traditional setup, programs are developed based on rules. Devs receive a set of rules, they convert them into code and there - we have a software. Imagine developers paving way for input data to take it to the expected outcome.

Concepts of ML and ANN come into picture when the requirement doesn't have defined rules. For example, predicting weather or identifying potential fraud in financial data. We rather train models with huge amount of data. Imagine data(like water) creating its own path to get to the best possible outcome.

Model is nothing but a set of flexible inter-connected knobs that we can tune in order to deliver the best possible result in any case.

This code is my attempt at understanding a typical training cycle (epoch) in a neural network with below mentioned concepts without going into metrices yet.  [`code/ANN_single_epoch_simulation.py`](code/ANN_single_epoch_simulation.py).
- Front propagation (weighted sums + biases, sigmoid)
- Loss calculation (squared error) and 
- Back propagation (partial derivation, chain rule, gradient descent)  

### Vector-Matrix multiplication & Matrix-Matrix multiplication

The previous section gave idea of a simple instance of a typical training process. There were two inputs with single hidden layer with two neurons.

Off course we need more layers and more neurons to make it capable of finding patterns in unorganized data.

To represent and calculate billions of values, metrices come handy for its convenience and efficiency that they unlock when used with GPUs due to parallel processing abilities.

Matrix multiplication steps are not that hard, however understanding them from core is a bit tricky. Here is my idea of it:

B = A @ X where Band A are metrices and X is an input vector; @ is multiplication.

In this example, A works as an actor and X is a subject. A will transform X into B.

We can imagine this from other pov as well where A and X are both transformers and we are proactively calculating its combined effect and storing it in B to use it in future on some input.

Let's say A and X both are our actors waiting for some input Y in future.
So we can say: (A @ X) @ Y = A @ (X @ Y).
Sequence does matter.

How does this apply in neural networks?
Imagine a network with 2 inputs and a hidden layer with 5 neurons in it.
there will be 10 weights in total which can be presented with 5x2 matrix W.

let's say input matrix 2x1 is called X, so weighted sum can be simply calculated with W @ X. Also note how 2 input values were moved forward and converted into 5 values.


If input vector was a matrix rather than a vector, we can say we are doing the same process for multiple input sets and process follows the same way.

