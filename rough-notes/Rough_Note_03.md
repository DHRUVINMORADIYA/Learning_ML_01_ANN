---
title: Rough Note 03
parent: Rough Notes
nav_order: 3
---

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

if we add softmax function in cross-entropy formula while getting delta l / delta oj, it beautifully boils down to y hat(j) - y(j). So I suppose this should be the only thing that matters in practical case. Rest seems important to develop math intuition.

- so summarizing this cross-entropy part
- formula is simple however going deeper into where it comes from gives better idea about the math in action.
- 2 ways to see it.
- first, maximum likelihood estimation - we want to maximize the likelihood of label Y when feaure X are given. As they are mutually independent scenarios we do multiplication -> for many multiplication, it would keep going near zero -> so we introduce log. log(ab) = log a + log b. -> log (yi | xi) is y hat j -> multiplying them with y (one-hot encoding) gives cross entropy formula.

- second, entropy and information theory - log value of a predicted probability gives surprisal value of any given prediction. the more it is near 1, we are less suprised. but if an event with 1% chance occurs, we are more surpised. log does exactly that -> multiplying them with their own prediction and summing them up gives us weighted average. multiplication because although some events might have high entropy but it doesn't occur all the time, so multiplying it with its initial prediction value gives its part in total entropy. and that becomes entropy function. -> In the case of classification model where we know real labels and their probability, we can do this multiplication with actual labels and that gives up cross-entropy formula.


-----------------------
some practicals start

Fashion-MNIST dataset classification

