Chapter 6: Linear Model Selection and Regularisation
================
Alfredo Hernández -
10 April 2018

# Exercise 3

Suppose we estimate the regression coefficients in a linear regression
model by minimising

$$
\\sum\_{i=1}^{n} \\left( y\_{i} -\\beta\_{0} -\\sum\_{j=1}^{p}\\beta\_{j}x\_{ij} \\right)\\quad \\text{subject to}\\quad \\sum\_{j=1}^{p}\\vert \\beta\_{j}\\vert \\leq s
$$

for a particular value of *s*. For parts (a) through (e), indicate which
of i. through v. is correct. Justify your answer.

1.  As we increase *s* from 0, the training RSS will:

<!-- -->

1.  Increase initially, and then eventually start decreasing in an
    inverted U shape.
2.  Decrease initially, and then eventually start increasing in a U
    shape.
3.  Steadily increase.
4.  Steadily decrease.
5.  Remain constant.

<!-- -->

1.  Repeat (a) for test RSS.

2.  Repeat (a) for variance.

3.  Repeat (a) for (squared) bias.

4.  Repeat (a) for the irreducible error.

<!-- ## Answers -->

## Exercise 3 (a)

1.  Steadily decreases: as *s* increases 0, all *β* increase from 0 to
    their least square estimate values. Therefore, the training error
    steadily decreases to the Ordinary Least Square RSS for the training
    data.

## Exercise 3 (b)

1.  Decrease initially, and then eventually start increasing in a U
    shape: when *s* = 0, all *β* are 0, the model has a high test RSS.
    As *s* increases, the *b**e**t**a* coefficients assume non-zero
    values and the model fits the test data better, resulting in a
    decrease of the test RSS. In the long run, the *b**e**t**a*
    coefficients start to overfit the training data, resulting in an
    increase of the test RSS.

## Exercise 3 (c)

1.  Steadily increase: variance always increases with fewer constraints.
    At *s* = 0, the model predicts a constant and has almost no
    variance. As *s* increases, the values of the different *β* become
    highly dependent on training data, resulting in an increase of the
    variance.

## Exercise 3 (d)

1.  Steadily decrease: bias always decreases with more model
    flexibility. WAt *s* = 0, the model predicts a constant and the
    prediction is far from actual value; thus the bias is high. As *s*
    increases, more *β* become non-zero and thus the model continues to
    fit training data better, resulting in a decrease of the bias.

## Exercise 3 (e)

1.  Remains constant: by definition, the irreducible error is
    independent of the selected model.
