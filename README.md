### Why do we need Artificial Neural Networks?
    
In traditional setup, programs are developed based on rules. Devs receive a set of rules, they convert them into code and there - we have a software. Imagine developers paving way for input data to take it to the expected outcome.

Concepts of ML and ANN come into picture when the requirement doesn't have defined rules. For example, predicting weather or identifying potential fraud in financial data. We rather train models with huge amount of data. Imagine data(like water) creating its own path to get to the best possible outcome.

Model is nothing but a set of flexible inter-connected knobs that we can tune in order to deliver the best possible result in any case.

This code is my attempt at understanding a typical training cycle (epoch) in a neural network with below mentioned concepts without going into metrices yet.  [`code/ANN_single_epoch_simulation.py`](code/ANN_single_epoch_simulation.py).
- Front propagation (weighted sums + biases, sigmoid)
- Loss calculation (squared error) and 
- Back propagation (partial derivation, chain rule, gradient descent)  
