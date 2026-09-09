Linear Neural Networks for Classification

Softmax - needed when we need probability destribution.
we use e**x to handle negatives and it also amplifies the differences between numbers.
we normalize values by taking proportional values of total sum of exponential functions.

cross-entropy formula: 

- y (log y hat)

two ways to derive it
1. taking sum of real values multiplied by log of expected value. why log -> it gives spread out spectrum for values input values between 0 and 1 giving higher value for lower inputs and slowly reducing it to 0 as we reach 1. simple case when batch size = 1.

2. maximum likelihood estimation. this is a way of probability where we are combining logs of multiple predicted outputs. see this as way to find loss when batch size > 1.

further intuition development of entropy and cross entropy.

Entropy is quantification of chaos. in this instance, it says how much the probablity is accumulated at one place or how much it is scattered. Ex. [1, 0, 0] has 0 entropy while [1/3,1/3,1/3] has highest entropy as there is no certainty at all.

oj (log oj) would give entropy of single element out of a vector. we do multiplication with oj to give weights to each surprisal. doing sum of of weighted entropis of all elements would give average entropy.

for cross-entropy, we replace multiplication part. we take real probability (coming from labels) to multiply with entropy of predicted probability.