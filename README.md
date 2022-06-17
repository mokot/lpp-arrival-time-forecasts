# Arrival time forecast

**Linear regression** is a basic and commonly used type of predictive analysis.  The 
overall idea of regression is to examine two things: *(1)* does a set of predictor 
variables do a good job in predicting an outcome (dependent) variable?  
*(2)* Which variables in particular are significant predictors of the outcome 
variable, and in what way do they–indicated by the magnitude and sign of the 
beta estimates–impact the outcome variable?  These regression estimates are used 
to explain the relationship between one dependent variable and one or more 
independent variables. The simplest form of the regression equation with one 
dependent and one independent variable is defined by the formula *y = c + b\*x*, 
where *y = estimated dependent variable score*, *c = constant*, *b = regression 
coefficient*, and *x = score on the independent variable*.

The **Gaussian processes model** is a probabilistic supervised machine learning 
framework that has been widely used for regression and classification
tasks. A Gaussian processes regression (GPR) model can make predictions 
incorporating prior knowledge (kernels) and provide uncertainty measures over predictions.

**Gradient boosting** is a machine learning technique used in regression and 
classification tasks, among others. It gives a prediction model in the form of 
an ensemble of weak prediction models, which are typically decision trees. When 
a decision tree is the weak learner, the resulting algorithm is called 
gradient-boosted trees; it usually outperforms random forest. A gradient-boosted 
trees model is built in a stage-wise fashion as in other boosting methods, 
but it generalizes the other methods by allowing optimization of an arbitrary 
differentiable loss function. 

Maximum accuracy was achieved using distributed gradient boosting, with which we 
erred by an average of **151.4** seconds. The second most successful approach was the 
Gaussian processes model approach, with which we made an average of **153.6** seconds of error. With 
the linear regression technique, which proved to be the worst for a given 
problem domain, we made an average of **154.0** seconds of error.