# output-boosting

The idea is simple: train a linear model with logistic regression, then add the softmax outputs to the features and train another linear model, etc.

# Results

The classifier doesn't work much better than a linear classifier on MNIST. The algorithm solves XOR, but that's an easy problem.
