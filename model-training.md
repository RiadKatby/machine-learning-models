## Importance of How Model Trained
Having a good understanding of training work help us in
- Select appropriate model quickly and pretty confidently
- Chose the right training algorithm to use
- Tune a good set of hyperparameters for the task. 
- Htlp debug issues and perform error analysis more efficiently
- Lastly, most of the topics discussed here will be essential in understanding, building, and training neural networks

## Agenda
- Closed-Form equation to traing linear model
- Iterative way to traing linear models Batch GD, Mini-batch GD, and Stochastic GD
- Polynomial Regression a complex model for non-linear datasets
- How to detect whether or not the model overfit using learning curves
- Discover some regularization techniques that can resuce the risk of overfitting
- Classification Models: Logistic and Softmax Regression

## Linear Regression model
We will study 2 different ways to train one of the simplest models
- Closed-Form equation
  Directly compute model parameters that best fit the model training set (model parameters that minimize the cost function)
- Iterative Optimization Approach (Gradient Descent)
  * Gradually tweaks model parameters to minimize cost function over training set
  * Converge same set of parameters as in Closed-Form
  * Look at variants Batch GD, Mini-batch GD, and Stochastic GD

## Linear Regression
### What is Linear Regiression Model?

- We saw previously life_satisfaction = θ0 + θ1 × GDP_per_capita
  * This model is just linear function of input features GDP_per_capita
  * θ0 and θ1 are the model's parameters
- Generally, Linear model making predictions by computing weighted sum of input features, plus constant called bias (intercept term)
  * y = θ0 + θ1x1 + θ2x2 + ⋯ + θnxn
	• ŷ is the predicted value.
	• n is the number of features.
	• xi is the ith feature value.
	• θj is the jth model parameter (including the bias term θ0 and the feature weights θ1, θ2, ⋯, θn).
  * y = hθ x = θ · x
	• θ is the model’s parameter vector, containing the bias term θ0 and the feature weights θ1 to θn.
	• x is the instance’s feature vector, containing x0 to xn, with x0 always equal to 1.
	• θ·x is the dot product of the vectors θ and x, which is of course equal to θ0x0 + θ1x1 + θ2x2 + ⋯ + θnxn.
	• hθ is the hypothesis function, using the model parameters θ.

## Linear Regression
### Vectors in Machine Learning

- Vectors are represented as column, 2D arrays with a single column.
- If θ and x are column vectors, then the prediction is: y = θTx
  * where θT is the transpose of θ (a row vector)
  * θTx is the matrix multiplication of θT and x.
  * Results same prediction, but represented as a single cell matrix rather than a scalar value.

## Linear Regression
### How do we train it?

- Training Model means settings its parameters in a way that best fits the training set.
- RMSE is common measure of how well or poor regression model fit the training sets
- It is easier find θ that minimize MSE than RMSE and it lead to the same result
- $` MSE(X, h_θ) = \frac{1}{m} \sum_{i=1}^m \left( θ^T x_i − y^\left(i \right) \right)^2 `$
