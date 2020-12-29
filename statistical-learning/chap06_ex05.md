Chapter 6: Linear Model Selection and Regularisation
================
Alfredo Hernández -
10 April 2018

# Exercise 5

It is well-known that Ridge regression tends to give similar coefficient
values to correlated variables, whereas the Lasso may give quite
different coefficient values to correlated variables. We will now
explore this property in a very simple setting.

Suppose that *n* = 2, *p* = 2, *x*<sub>11</sub> = *x*<sub>12</sub>,
*x*<sub>21</sub> = *x*<sub>22</sub>. Furthermore, suppose that
*y*<sub>1</sub> + *y*<sub>2</sub> = 0 and
*x*<sub>11</sub> + *x*<sub>21</sub> = 0 and
*x*<sub>12</sub> + *x*<sub>22</sub> = 0, so that the estimate for the
intercept in a least squares, Ridge regression, or Lasso model is zero:
*β̂*<sub>0</sub> = 0.

1.  Write out the Ridge regression optimisation problem in this setting.

2.  Argue that in this setting, the Ridge coefficient estimates satisfy
    *β̂*<sub>1</sub> = *β̂*<sub>2</sub>.

3.  Write out the Lasso optimisation problem in this setting.

4.  Argue that in this setting, the Lasso coefficients *β̂*<sub>1</sub>
    and *β̂*<sub>2</sub> are not unique; in other words, there are many
    possible solutions to the optimisation problem in (c). Describe
    these solutions.

<!-- ## Answers -->

## Exercise 5 (a)

A general form of Ridge regression optimisation looks like
$$
\\text{Minimise:} \\quad 
\\sum\\limits\_{i=1}^{n} {(y\_i - \\hat{\\beta}\_0 - \\sum\\limits\_{j=1}^{p} {\\hat{\\beta}\_jx\_j} )^2} + \\lambda \\sum\\limits\_{j=1}^{p} \\hat{\\beta}\_{j}^{2} .
$$

In this case, *β̂*<sub>0</sub> = 0 and *n* = *p* = 2. So, the
optimisation looks like:
$$
\\boxed{
\\text{Minimise:} \\quad
(y\_1 - \\hat{\\beta}\_1x\_{11} - \\hat{\\beta}\_2x\_{12})^2 + (y\_2 - \\hat{\\beta}\_1x\_{21} - \\hat{\\beta}\_2x\_{22})^2 + \\lambda (\\hat{\\beta}\_1^2 + \\hat{\\beta}\_2^2)
}.
$$

## Exercise 5 (b)

First of all, we set
*x*<sub>11</sub> = *x*<sub>12</sub> = *x*<sub>1</sub> and
*x*<sub>21</sub> = *x*<sub>22</sub> = *x*<sub>2</sub>. Then, we expand
the previous expression; and we take the partial deritive to
*β̂*<sub>1</sub> and set equation to 0:
$$
(\\hat{\\beta}\_1x\_1^2-x\_1y\_1+\\hat{\\beta}\_2x\_1^2) + (\\hat{\\beta}\_1x\_2^2-x\_2y\_2+\\hat{\\beta}\_2x\_2^2) + \\lambda\\hat{\\beta}\_1 = 0
\\\\
\\Rightarrow \\hat{\\beta}\_1 (x\_1^2+x\_2^2) + \\hat{\\beta}\_2 (x\_1^2+x\_2^2) + \\lambda\\hat{\\beta}\_1 = x\_1y\_1 + x\_2y\_2 .
$$

Now we add 2*β̂*<sub>1</sub>*x*<sub>1</sub>*x*<sub>2</sub> and
2*β̂*<sub>2</sub>*x*<sub>1</sub>*x*<sub>2</sub> to both sides of the
equation:

$$
\\hat{\\beta}\_1 (x\_1^2 + x\_2^2 + 2x\_1x\_2) + \\hat{\\beta}\_2 (x\_1^2 + x\_2^2 + 2x\_1x\_2) + \\lambda\\hat{\\beta}\_1 
= x\_1y\_1 + x\_2y\_2 + 2\\hat{\\beta}\_1x\_1x\_2 + 2\\hat{\\beta}\_2x\_1x\_2 
\\\\
\\hat{\\beta}\_1 (x\_1 + x\_2)^2 + \\hat{\\beta}\_2 (x\_1 + x\_2)^2 + \\lambda\\hat{\\beta}\_1 
= x\_1y\_1 + x\_2y\_2 + 2\\hat{\\beta}\_1x\_1x\_2 + 2\\hat{\\beta}\_2x\_1x\_2 , 
$$

and because *x*<sub>1</sub> + *x*<sub>2</sub> = 0, we can eliminate the
first two terms:

*λ**β̂*<sub>1</sub> = *x*<sub>1</sub>*y*<sub>1</sub> + *x*<sub>2</sub>*y*<sub>2</sub> + 2*β̂*<sub>1</sub>*x*<sub>1</sub>*x*<sub>2</sub> + 2*β̂*<sub>2</sub>*x*<sub>1</sub>*x*<sub>2</sub>.

Likewise by taking the partial deritive to *β̂*<sub>2</sub>, we can get
the equation:
*λ**β̂*<sub>2</sub> = *x*<sub>1</sub>*y*<sub>1</sub> + *x*<sub>2</sub>*y*<sub>2</sub> + 2*β̂*<sub>1</sub>*x*<sub>1</sub>*x*<sub>2</sub> + 2*β̂*<sub>2</sub>*x*<sub>1</sub>*x*<sub>2</sub>.

Finally, we see that the left side of the equations for both
*λ**β̂*<sub>1</sub> and *λ**β̂*<sub>2</sub> are the same so we have:

$$
\\lambda\\hat{\\beta}\_1 = \\lambda\\hat{\\beta}\_2 
\\Rightarrow 
\\boxed{
\\hat{\\beta}\_1 = \\hat{\\beta}\_2
}.
$$

## Exercise 5 (c)

A general form of Lasso optimisation looks like

$$
\\text{Minimise:} \\quad 
\\sum\\limits\_{i=1}^{n} {(y\_i - \\hat{\\beta}\_0 - \\sum\\limits\_{j=1}^{p} {\\hat{\\beta}\_jx\_j} )^2} + \\lambda \\sum\\limits\_{j=1}^{p} \| \\hat{\\beta\_{j}} \|.
$$

For Lasso, like in Ridge regression, we have

$$
\\boxed{
\\text{Minimise:} \\quad
(y\_1 - \\hat{\\beta}\_1x\_{11} - \\hat{\\beta}\_2x\_{12})^2 + (y\_2 - \\hat{\\beta}\_1x\_{21} - \\hat{\\beta}\_2x\_{22})^2 + \\lambda (\| \\hat{\\beta}\_1 \| + \| \\hat{\\beta}\_2 \|)
}.
$$

## Exercise 5 (d)

Here is a geometric interpretation of the solutions for the equation in
(c) above. We use the alternate form of Lasso constraints
\|*β̂*<sub>1</sub>\| + \|*β̂*<sub>2</sub>\| &lt; *s*.

The Lasso constraint takes the form
\|*β̂*<sub>1</sub>\| + \|*β̂*<sub>2</sub>\| &lt; *s*, which when plotted
takes the familiar shape of a diamond centered at origin (0, 0).

Next consider the squared optimisation constraint
(*y*<sub>1</sub> − *β̂*<sub>1</sub>*x*<sub>11</sub> − *β̂*<sub>2</sub>*x*<sub>12</sub>)<sup>2</sup> + (*y*<sub>2</sub> − *β̂*<sub>1</sub>*x*<sub>21</sub> − *β̂*<sub>2</sub>*x*<sub>22</sub>)<sup>2</sup>.
We use the facts *x*<sub>11</sub> = *x*<sub>12</sub>,
*x*<sub>21</sub> = *x*<sub>22</sub>,
*x*<sub>11</sub> + *x*<sub>21</sub> = 0,
*x*<sub>12</sub> + *x*<sub>22</sub> = 0 and
*y*<sub>1</sub> + *y*<sub>2</sub> = 0 to simplify it to

Minimise:  2(*y*<sub>1</sub> − (*β̂*<sub>1</sub> + *β̂*<sub>2</sub>)*x*<sub>11</sub>)<sup>2</sup>.

This optimisation problem has a simple solution:
$\\hat{\\beta}\_1 + \\hat{\\beta}\_2 = \\frac{y\_1}{x\_{11}}$. This is a
line parallel to the edge of Lasso-diamond
*β̂*<sub>1</sub> + *β̂*<sub>2</sub> = *s*. Now solutions to the original
Lasso optimisation problem are contours of the function
(*y*<sub>1</sub> − (*β̂*<sub>1</sub> + *β̂*<sub>2</sub>)*x*<sub>11</sub>)<sup>2</sup>
that touch the Lasso-diamond *β̂*<sub>1</sub> + *β̂*<sub>2</sub> = *s*.

Finally, as *β̂*<sub>1</sub> and *β̂*<sub>2</sub> vary along the line
$\\hat{\\beta}\_1 + \\hat{\\beta}\_2 = \\frac{y\_1}{x\_{11}}$, these
contours touch the Lasso-diamond edge
*β̂*<sub>1</sub> + *β̂*<sub>2</sub> = *s* at different points. As a
result, the entire edge *β̂*<sub>1</sub> + *β̂*<sub>2</sub> = *s* is a
potential solution to the Lasso optimisation problem.

A similar argument can be made for the opposite Lasso-diamond edge:
*β̂*<sub>1</sub> + *β̂*<sub>2</sub> =  − *s*.

Thus, the Lasso problem does not have a unique solution. The general
form of solution is given by two line segments:

$$
\\boxed{
\\hat{\\beta}\_1 + \\hat{\\beta}\_2 = s, \\quad \\hat{\\beta}\_1 \\geq 0, \\, \\hat{\\beta}\_2 \\geq 0
},
\\\\
\\text{and}
\\\\
\\boxed{
\\hat{\\beta}\_1 + \\hat{\\beta}\_2 = -s, \\quad \\hat{\\beta}\_1 \\leq 0, \\,  \\hat{\\beta}\_2 \\leq 0
}.
$$
