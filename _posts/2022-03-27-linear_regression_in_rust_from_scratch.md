---
layout:     post
title:      "Ordinary least squares linear regression in Rust"
date:       2022-03-27 10:30:00 +0100
categories: ["Machine Learning", "Rust"]
---

* TOC
{:toc}

# Introduction

In contrast to the widespread use of Python and common machine learning packages like scikit-learn [[1]](#r1), there is an advantage in doing things from scratch. Namely, learning how things understand gives you an advantage in choosing the right algorithms for the job later down the line. We will start doing that with the most simple machine learning algorithm and maybe the most commonly used one, linear regression. In this article we are going to implement the so called ordinary least squares (OLS) [[2]](#r2) linear regression [[3]](#r3) in Rust [[4]](#r4). We will show that with just a few lines of code it is possible to implement it from scratch and using it for some simple modelling. During this work we will gain a better understanding of the concept, the algorithm and its limits, and we learn about the Rust package called nalgebra [[5]](#r5), which will help us with our linear algebra needs.

# Linear Regression

## What is linear regression

Linear regression [[3]](#r3) is used to model the relationship between a response/target variable and (multiple) explanatory variables or parameters. It is called linear, because the coefficients in the model are linear. A linear model for the target variable $y_i$ can be written in the form

$$  y_i = \beta_{0} + \beta_{1} x_{i1} + \cdots + \beta_{p} x_{ip} + \varepsilon_i \, ,
 \qquad i = 1, \ldots, n \; . $$

where $x_{ip}$ are the explanatory variables and $\beta_{p}$ are unknown coefficients. The $\varepsilon_i$ are called error terms or noise and they capture all the other information that we cannot model with this linear approach.

One can also write this in matrix form as

$$ \mathbf{y} = X\boldsymbol\beta + \boldsymbol\varepsilon, \,$$

where all the $n$ equations are squashed together. As we will see below, this notation is useful for deriving our method of determining the parameters $\boldsymbol\beta$. Note: Here we integrated the $\beta_0$ in the $\boldsymbol\beta$ and therefore $\boldsymbol\beta$ is now a $(p+1)$-dimensional vector and $\mathbf{x}$ is now a (n, p+1)-dimensional matrix, where we include a constant as first column, e.g., $x_{i0}=1$ for $i = 1, \ldots, n$.

## Derivation of ordinary least squares linear regression

Ordinary least squares (OLS) [[2]](#r2) is, as the name suggests, a least squares method for find the unknown parameters for a linear regression model. The idea is to minimize the sum of squares of the differences between the observed target variable and the predicted target variable using the linear regression model.

For the linear case the minimization problems possesses a unique global minimum and its solution can be expressed by an explicit formula for the coefficients $\boldsymbol\beta$:

$$ \boldsymbol\beta = (\mathbf{x}^\mathbf{T}\mathbf{x})^{-1}\mathbf{y} $$

// TODO: Describe some useful metrics, like R^2, mean square errors

# Implementation in Rust

Before we start, we need to bring in the nalgebra functions that we are going to use
```rust
extern crate nalgebra as na;
use std::ops::Mul;

use na::{DMatrix, DVector};
```

Then we define the x values and the y values that we want to fit
```rust
let x_training_values = na::dmatrix![
    1.0f64, 3.0f64;
    2.0f64, 1.0f64;
    3.0f64, 8.0f64;
];
let y_values = na::dvector![2.0f64, 3.0f64, 4.0f64];
```

Fitting the model can be performed as follows:
```rust
w = x_values
    .tr_mul(&x_values)
    .try_inverse()
    .unwrap()
    .mul(x_values.transpose())
    .mul(y_values);
```
This also tries to calculate the inverse of the $X^TX$ matrix. As this can fail, if the matrix is not invertible (for example because the columns in $x$ are not linearly independent), in a production environment one should handle this gracefully and don't use unwrap. For our coding example, however, it is good enough.

Getting predictions is then as simple as
```rust
let prediction = x_values.mul(w);
```

The full source code for LinearRegression can be seen here (excuse the unwraps). It contains a bit more boilerplate, but also support fitting without the intercept term, which can be useful if the data is already centered around its mean (in which case the intercept would be zero).
```rust
use std::ops::Mul;

use na::{DMatrix, DVector};

/// Ordinary least squares linear regression
///
/// It fits a linear model of the form y = b_0 + b_1*x + w_2*x_2 + ...
/// which minimizes the residual sum of squared between the observed targets
/// and the predicted targets.
pub struct LinearRegression {
    w: Option<DVector<f64>>,
    fit_intercept: bool,
}

impl LinearRegression {
    /// Returns a linear regressor using ordinary least squares linear regression
    ///
    /// # Arguments
    ///
    /// * `fit_intercept` - Whether to fit a intercept for this model.
    ///     If false assume that the data is centered, i.e., intercept is 0.
    ///
    pub fn new(fit_intercept: bool) -> LinearRegression {
        LinearRegression {
            w: None,
            fit_intercept,
        }
    }

    /// Fit the model
    ///
    /// # Arguments
    ///
    /// * `x_values` - parameters of shape (n_samples, n_features)
    /// * `y_values` - target values of shape (n_samples)
    pub fn fit(&mut self, x_values: &DMatrix<f64>, y_values: &DVector<f64>) {
        if self.fit_intercept {
            let x_values = x_values.clone().insert_column(0, 1.0);
            self._fit(&x_values, y_values);
        } else {
            self._fit(x_values, y_values);
        }
    }
    fn _fit(&mut self, x_values: &DMatrix<f64>, y_values: &DVector<f64>) {
        self.w = Some(
            x_values
                .tr_mul(&x_values)
                .try_inverse()
                .unwrap()
                .mul(x_values.transpose())
                .mul(y_values),
        );
    }
    pub fn intercept(&self) -> Result<f64, String> {
        if !self.w.is_some() {
            return Err("Model was not fitted".to_string());
        }
        if self.fit_intercept {
            return Ok(self.w.as_ref().unwrap()[0]);
        }
        return Err("Model was not fitted with intercept".to_string());
    }

    /// Returns the predictions using the provided parameters `x_values`
    ///
    /// # Arguments
    ///
    /// * `x_values` - parameters of shape (n_samples, n_features)
    pub fn predict(&self, x_values: &DMatrix<f64>) -> Result<DVector<f64>, String> {
        if let Some(w) = self.w.as_ref() {
            if self.fit_intercept {
                let x_values = x_values.clone().insert_column(0, 1.0);
                let res = x_values.mul(w);
                return Ok(res);
            } else {
                let res = x_values.mul(w);
                return Ok(res);
            }
        }
        Err("Model was not fitted.".to_string())
    }
}
```

I will update this blog post when I decide to upload the code to my GitHub page.

# Example: Diabetes dataset

// TODO: Develop and describe diabetes dataset regression based on https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html

# References

[1]<a name="r1"></a> [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)  
[2]<a name="r2"></a> [https://en.wikipedia.org/wiki/Ordinary_least_squares](https://en.wikipedia.org/wiki/Ordinary_least_squares)  
[3]<a name="r3"></a> [https://en.wikipedia.org/wiki/Linear_regression](https://en.wikipedia.org/wiki/Linear_regression)  
[4]<a name="r4"></a> [https://www.rust-lang.org/](https://www.rust-lang.org/)  
[5]<a name="r5"></a> [https://nalgebra.org/](https://nalgebra.org/)
