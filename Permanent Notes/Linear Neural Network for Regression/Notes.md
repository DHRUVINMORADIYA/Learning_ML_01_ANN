Summary

Linear Neural Network is the most primal form of neural networks. We can see it as a single neuron separated out from a bigger network. The complex neural networks will contain more of this all nicely linked together.

Described as a linear algebra function,

$$
\hat{y} = w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + b
$$

If used metrices,

$$
\hat{y} =
\begin{bmatrix} w_1 & w_2 & \cdots & w_n \end{bmatrix}
\begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}
+ b
$$

where x1, x2, ..., xn are input values and w1, w2, ..., xn are weights describing how much importance we are giving to each corresponding input value while coming up with prediction (ŷ); b is bias. With bias we give a bit of firm identity to each perticular neuron. We assign each neuron its own parameter which is not dependent on any other neuron.

Above function can be viewed as a vector on n-dimensional space.

