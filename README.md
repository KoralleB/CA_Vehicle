# Powertrain Prediction Using Machine Learning Models

## Background
This is a machine learning project prepared for the graduate-level course Statistical Methods in Machine Learning STA208 at UC Davis.

The dataset used for the project  is the [California Vehicle Survey](https://www.energy.ca.gov/data-reports/surveys/california-vehicle-survey).

## Table of Content
1. [Abstract](#abstract)
2. [Scripts](#scripts)
3. [Data Exploratory Analysis](#data-exploratory-analysis)
4. [Modeling and Results](#modeling-and-results)
5. [Policy Implementation](#policy-implementation)
6. [Conclusions](#conclusions)
7. [References](#references)
8. [Tech Stack](#tech-stack)

## Abstract
In this project, our objective is to classify vehicles by powertrain type using personal and household level data from
the 2019 California Vehicle Survey. Two powertrain types are considered: internal combustion engine vehicle (ICEV)
represents gasoline, diesel and other traditional fuel types, and electric vehicle (EV), represents alternative fuel
vehicles, such as battery electric vehicle, plug-in hybrid electric vehicle, and more. 
  
In particular, we wish to compare the performance of standard Logistic Regression
models to other classification models typically seen in Machine learning to evaluate the potential use of these models 
in future research. In addition to logistic regression (log) models, Random Forests (rf), K-Nearest Neighbors (knn) 
algorithm, and Neural Network (nn) are considered. Class imbalance present in the data poses a potential problem for predictive 
performance. Consequently, adjustments are made and the model performances are investigated both with and without 
adjustments (oversampling) to see how models handle the class imbalance, and which benefit the most from attempts to 
correct for the imbalance.

Highly performed classification methods can benefit policy makers and planners to anticipate the energy demand and 
required infrastructure in California raised by electric vehicle users.

## Scripts
1. [data_prepro.py](https://github.com/KoralleB/CA_Vehicle/blob/master/code/data_prepro.py) - preprocessing and cleaning the three datasets, and merging into one.
2. [data_plot.py](https://github.com/KoralleB/CA_Vehicle/blob/master/code/data_plot.py) - plotting functions for data exploration.
3. [data_split.py](https://github.com/KoralleB/CA_Vehicle/blob/master/code/data_split.py) - split dataset into train and test X and y, and resample.
4. [model_logit.py](https://github.com/KoralleB/CA_Vehicle/blob/master/code/model_logit.py) - cross-validate hyperparameter, fit, predict, output performance variables, and model with best hyperparameters for Logistic Regression.
5. [model_rf.py](https://github.com/KoralleB/CA_Vehicle/blob/master/code/model_rf.py) - cross-validate hyperparameter, fit, predict, output performance variables, and model with best hyperparameters for Random Forests.
6. [model_knn.py](https://github.com/KoralleB/CA_Vehicle/blob/master/code/model_knn.py) - cross-validate hyperparameter, fit, predict, output performance variables, and model with best hyperparameters for K Nearest Neighbors.
7. [model_nn.py](https://github.com/KoralleB/CA_Vehicle/blob/master/code/model_nn.py) - define and train model, predict, output accuracy, confusion matrix, roc, pr, and training arrays Neural Nets.
8. [model_plot.py](https://github.com/KoralleB/CA_Vehicle/blob/master/code/model_plot.py) - plotting functions for model performance.
9. [map_plot.py](https://github.com/KoralleB/CA_Vehicle/blob/master/code/map_plot.py) - plotting functions for mapping true and predicted results.

## Data Exploratory Analysis
Please see the [Data Exploratory Analysis notebook](https://github.com/KoralleB/CA_Vehicle/blob/master/notebook/DataExplore.ipynb).
The notebook includes data exploration, including explanation about the data and variable choice reasoning. 

## Modeling and Results
Please see the [Modeling and Results notebook](https://github.com/KoralleB/CA_Vehicle/blob/master/notebook/Performance.ipynb).
The notebook includes performance evaluation of different resampling techniques for each model, with a similar evaluation 
for different Neural Net architectures. The notebook discusses methods and conclusions.

For each model and balancing technique, we plotted ROC curve, PR Curve, and confusion matrix. 
Here is a snapshot of ROC and PR curves of Random Forests model, wihtout balancing, and with oversampling smote balancing techniques.
<img src='https://github.com/KoralleB/CA_Vehicle/blob/master/image/ROC%20rf.PNG' width='400'> <img src='https://github.com/KoralleB/CA_Vehicle/blob/master/image/PR%20rf.PNG' width='400'>

## Policy Implementation
Please see the [Policy Implementation notebook](https://github.com/KoralleB/CA_Vehicle/blob/master/notebook/MapPlots.ipynb).
The notebook includes true and prediction results visualized with CA maps. Notebook discusses final conclusions and policy implementation.

## Conclusions
No model appears to significantly outperform all others by all standards of model evaluation. However the diversity
of strengths and weaknesses across models suggests the potential for future application of machine learning methods
for transportation choice modeling.

As of the task of predicting vehicle ownership by powertrain type, it is important for policymakers to work with 
models that maximize the true prediction rate of EV while minimizing the EV false predictions, to avoid the waste of 
state budget for mega infrastructure projects related to the electricity grid. With that in mind, we see that the 
traditional logistic regression model highly overestimates the EV share in every California region, while the ML 
algorithms predict rates of EV share that are closer to the true rate. We conclude that there is room for machine 
learning modeling in this area, and the investment in such tools can be beneficial in the long-term.

## References
* J. Jia, “Analysis of Alternative Fuel Vehicle (AFV) Adoption Utilizing Different Machine Learning Methods: 
A Case Study of 2017 NHTS,” IEEE Access, vol. 7, pp. 112726–112735, 2019, doi: 10.1109/access.2019.2934780.
* W. Li and K. M. Kockelman, “How does machine learning compare to conventional econometrics for transport data sets? 
A test of ML vs MLE,” Transp. Res. Rec., 2020, doi: 10.1017/CBO9781107415324.004.
* T. Hastie, R. Tibshirani, and J. Friedman, “The Elements of Statistical Learning,” Second edition, Corrected 12th printing, Springer.
* B. Bae and H.-L. Hwang, “Predicting Daily Trip Frequencies of Vulnerable Households in New York State Using 
Emerging Machine-Learning Approaches,” 2018.

The project was prepared by Koral Buch and Kenneth Broadhead.

## Tech Stack
<img src='https://github.com/KoralleB/CA_Vehicle/blob/master/image/logo_NumPy.png' width='400'> <img src='https://github.com/KoralleB/CA_Vehicle/blob/master/image/logo_pandas.png' width='400'>

<img src='https://github.com/KoralleB/CA_Vehicle/blob/master/image/logo_plotly.png' width='400'> <img src='https://github.com/KoralleB/CA_Vehicle/blob/master/image/logo_matplotlib.png' width='400'>

<img src='https://github.com/KoralleB/CA_Vehicle/blob/master/image/logo_seaborn.png' width='400'> <img src='https://github.com/KoralleB/CA_Vehicle/blob/master/image/logo_plotnine.png'>

<img src='https://github.com/KoralleB/CA_Vehicle/blob/master/image/logo_tensorflow.png' width='400'> <img src='https://github.com/KoralleB/CA_Vehicle/blob/master/image/logo_scikit.png' width='400'>
