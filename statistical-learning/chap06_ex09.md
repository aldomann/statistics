Chapter 6: Linear Model Selection and Regularisation
================
Alfredo Hernández -
10 April 2018

# Exercise 9

In this exercise, we will predict the number of applications received
using the other variables in the `College` data set.

1.  Split the data set into a training set and a test set.

2.  Fit a linear model using least squares on the training set, and
    report the test error obtained.

3.  Fit a Ridge regression model on the training set, with *λ* chosen by
    cross-validation. Report the test error obtained.

4.  Fit a Lasso model on the training set, with *λ* chosen by
    cross-validation. Report the test error obtained, along with the
    number of non-zero coefficient estimates.

5.  Comment on the results obtained. How accurately can we predict the
    number of college applications received? Is there much difference
    among the test errors resulting from these five approaches?

<!-- ## Answers -->

## Exercise 9 (a)

First of all, we load the `ISLR` package to work with the `College` data
set:

``` r
library(ISLR)
```

Before doing anything, let’s make sure the data set does not have
missing data:

``` r
table(is.na(College))
```

    ## 
    ## FALSE 
    ## 13986

As we can see, the data is alright and there is no need to clean it.

Now we can work with the `College` data set and create a `college.train`
(50% of the data) and `college.test` (50% of the data set):

``` r
set.seed(10)
college.size <- nrow(College)
train.ind <- sample(1:college.size, college.size / 2)

college.train <- College[train.ind, ]
college.test <- College[-train.ind, ]
```

## Exercise 9 (b)

Now we will perform a ordinary least squares (OLS) linear model for the
number of applications (`Apps`) and calculate the RSS for the test data:

``` r
apps.lm <- lm(Apps ~ ., data = college.train)
apps.pred.lm <- predict(apps.lm, newdata = college.test)
apps.lm.test.rss <- mean((college.test$Apps - apps.pred.lm)^2)
```

The RSS for the test data using OLS is 1.0200998^{6}.

## Exercise 9 (c)

Now we want to perform a Ridge regression on the data:

``` r
xmat.train <- model.matrix(Apps ~ ., data = college.train)[, -1]
xmat.test <- model.matrix(Apps ~ ., data = college.test)[, -1]
```

We need the `glmnet` package to use Ridge regression on the data:

``` r
library(glmnet)
```

Ridge regression involves tuning a hyperparameter, lambda. `cv.glmnet()`
will generate default values, but it is common practice to define our
own with the `lambda` argument:

``` r
lambdas <- 10^seq(4, -2, by = -.1)
apps.ridge <- cv.glmnet(xmat.train, college.train$Apps, alpha = 0, lambda = lambdas)
best.lambda.ridge <- apps.ridge$lambda.min
```

Now we simply calculate the RSS for the test data:

``` r
apps.pred.ridge <- predict(apps.ridge, s = best.lambda.ridge, newx = xmat.test)
apps.ridge.test.rss <- mean((college.test$Apps - apps.pred.ridge)^2)
```

The RSS for the test data using a Ridge regression is 1.0200404^{6},
which is slightly lower than the one obtained using OLS.

## Exercise 9 (d)

Now we are going to perform a Lasso regression:

``` r
apps.lasso <- cv.glmnet(xmat.train, college.train$Apps, alpha = 1, lambda = lambdas)
best.lambda.lasso <- apps.lasso$lambda.min
```

Now we simply calculate the RSS for the test data:

``` r
apps.pred.lasso <- predict(apps.lasso, s = best.lambda.lasso, newx = xmat.test)
apps.lasso.test.rss <- mean((college.test$Apps - apps.pred.lasso)^2)
```

The RSS for the test data using a Lasso regression is 1.0079555^{6},
which is lower than both OLS and Ridge regressions.

## Exercise 9 (g)

Let us compare which coefficients are the “less important” according to
each of the methods used. First of all, let us analyse the OLS:

``` r
apps.lm$coefficients[abs(apps.lm$coefficients) < 0.05]
```

    ## F.Undergrad P.Undergrad 
    ##  0.01992043  0.04212868

For the shrinkage methods, we will use a custom function to get the
coefficients:

``` r
get_reduced_vars <- function(pred, value) {
  pred[, 1][abs(pred[, 1]) < value]
}
```

``` r
ridge.coeffs <- predict(apps.ridge, s = best.lambda.ridge, type = "coefficients")
lasso.coeffs <- predict(apps.lasso, s = best.lambda.lasso, type = "coefficients")
```

``` r
get_reduced_vars(ridge.coeffs, 0.05)
```

    ## F.Undergrad P.Undergrad 
    ##  0.01937588  0.04226025

``` r
get_reduced_vars(lasso.coeffs, 0.05)
```

    ##   Top25perc F.Undergrad P.Undergrad       Books    Personal perc.alumni 
    ## 0.000000000 0.000000000 0.004869013 0.037651466 0.016039646 0.000000000

As it seems, both OLS and Ridge reduce the `P.Undergrad` variable to a
low value, whilst Lasso totally shrinks it to zero, as well as the
`Books` variable.

Here are the test $R^2 = 1 - \\dfrac{RSS}{TSS}$ for all models:

``` r
test.avg <- mean(college.test$Apps)
apps.test.tss <- mean((college.test$Apps - test.avg)^2)
r2.lm.test <- 1 - apps.lm.test.rss / apps.test.tss
r2.ridge.test <- 1 - apps.ridge.test.rss / apps.test.tss
r2.lasso.test <- 1 - apps.lasso.test.rss / apps.test.tss
```

Let us see the *R*<sup>2</sup> for the three models:

``` r
apps.results <- tibble(
  method = as.factor(c("OLS", "Ridge", "Lasso")),
  r2 = c(r2.lm.test, r2.ridge.test, r2.lasso.test)
)
```

``` r
ggplot(apps.results) +
  geom_point(aes(x = method, y = r2)) +
  labs(title = "Test R-squared", x = "Method", y = "R-Squared")
```

<img src="figures/chap06_ex09-gg_apps_results-1.png" style="display: block; margin: auto;" />

The plot shows that test *R*<sup>2</sup> for all models are around 0.9,
with Lasso having slightly higher test *R*<sup>2</sup> than OLS and
Ridge. We can conclude that all models predict college applications with
high accuracy.
