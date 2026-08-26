Planning of maintenance of permanent notes

sequence
- MNIST program to understand layers, neurons, activation functions, idea of seeing patterns from a neuron's POV on high level
- visual geometry intution development of matrix multiplication (3blue1brown)
- doing a forward prop, loss calc and back prop cycle on paper (and later in python) to better see gradient descent
- start of book deep dive in deep learning to give linear path to learning.

contents of book understood till now
- starts with some theory

1. key components
data - data using which we train, validate and test models
model - structure/function that gives us predictions(y hat)
objective function - loss function - telling us magnitude of mismatch with expectation
algorithm - way we use to update the weights and biases

2. supervised learning model types [These parts were taken more like set of words and ideas on what to expect in future; I might come back to it time to time to connect dots on what they really meant.]
- regression - how many/how much - predicting a number - uses relu in general (predicting house price)
- classification - when output is within a set of values - uses sigmoid and cross entropy loss function is used in general (predict animal, MNIST)
    - binary
    - multiclass
    - hirarchical
- Tagging - multiple things present in single input
- Search - rank results by most relevant to least
- Recommender Systems - results depend on personal preferences given explicitely or implicitely
- Sequence learning - when data is not a single thing but a stream; knowing previous and next elements in necessary
    - tagging and parsing (inputs and outputs are aligned)
    - automatic speech recognition (inputs and outputs are not aligned)
    - text to speech
    - translation

3. unsupervised/ self-supervised learning intro

- clustering - group together similar things
- subspace estimation (principle component analysis if linear) - we reduce number of parameters (some connection with eigenvectors that is yet to learn)
- finding similar properties in high dimentional space (euclidian distance)
- casuality and probabilistic models - relates to cause-effect relationships
- deep generative models

4. reinforcement learning (f around and find out)
    markov decision process (general framework)
    contexual bandit problem (have context and based on that choose from options)
    multi-armed bandit problem (exploration - explore unknown terretory vs exploitation - select form best known option)


Then came data manupulation

- NumPy uses ndarray which uses CPU
- MXNet uuses ndarray which uses GPU
- PyTorch and TensorFlow uses Tensors which uses GPU (we will use PyTorch mostly)


linear algebra

Scaler(single digit) -> Vector(1D) -> Matrix(2D) -> Tensors(nD) - all can be saved and processed in tensor

- some matrix operations were covered. most of the ideas are simple so no need to add here. However, I might come back here to add anything new that comes under linear algebra.
- I felt Matrix multiplication important. It is important to understand it from geometric perspective and from its use case in neural networks. wrote down understanding in Rough_Note_01.md

calculus
- understanding of differentiation
- partial derivatives
- gradients
- chain rule
- automatic differentiation was new and felt important - you turn on a switch over a variable and in all subsequent steps gradient values are stored automatically, even when process is defined at runtime, we can trace back to gradients.
for example,
a.requires_grad_(True) # turning on switch
b = 2 * torch.dot(a,a) # creating formula, defining process
b.backward() # fills gradient values for all participating variables in formula
a.grad # gives partial derivative value of a (slope of a with reference to b)

probability and statistics
- Bayes' theorem
P(A|B) = P(B|A) x P(A) / P(B)
this is about understanding how to update what we believe about something after seeing new evidence

- there are bunch of notations and concepts that were covered. I am skipping them as I have not yet seen the connectin with ML. I might come back later here when I'm in need of some ground understanding.


then comes linear regression
- given that I already understood MLP, this one felt easy. listing some things that I learned.
1. works on simple function -> y = w1x1 + w2x2 + b
2. loss function - squared error
3. analytic solution w∗ = (XtX)**-1 Xt y
- as here we have linear equation here without any activation function, it is possible to compress back propagation process into a single math formula that will give us the best weights tuning possible to make the loss minimum. derivation comes from assigning zero to derivation function and doing some math. I'm not going into that as of now.
4. stochstic gradient descent - this one I have noted the notes in rough note 01. it is basically back prod process.
5. normal distribution (gaussian distribution) and how we assume noise that comes along with data follows gaussian distrubution
- "If the unexplained errors in our data behave like Gaussian noise, then the model parameters that make the observed data most probable are exactly the parameters that minimize squared error."
- "Most unexplained effects roughly cancel out, while large deviations are progressively less common."

above points were mainly theory
next we saw its implementation in python. 
- OOP for ML - we separate things into 3 classes - Model, Data and Trainer.
- Model contains all the processes - front prop, loss calc and back prop; data helps to supply data in batches; trainer is orchestrator that will coordinate the other two.
- Next we saw sythetic data generation. we can define standard deviation with mid point and it will follow gaussian distrubtion to generate train and test data.
- We saw code that did above things. This is nearest to implementing a linear regression by hand. However in real life, we might not go this deep into logic. we have libraries with all boiler plate code with best efficiency. [I will have to write and run a code for the same by myself.]

Next we did the same but here with the use of pre written libraries. code is really compressed into few lines in real life. but I feel investing time to understand inner architecture might prove out to fruitful in future. [I will have to write and run a code for the same by myself.]