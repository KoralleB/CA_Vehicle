# Vehicle Ownership - Powertrain Prediction

### Abstract
In this project, our objective is to classify vehicles by powertrain type using personal and household level data from
the 2019 California Vehicle Survey. In particular, we wish to compare the performance of standard Logistic Regression
models to other classification models typically seen in Machine learning to evaluate the potential use of these models 
in future research. In addition to logistic regression (log) models, Random Forests (rf), Nearest Neighbor (knn) algorithm, 
and Neural Network (nn) are considered. Class imbalance present in the data poses a potential problem for predictive 
performance. Consequently, adjustments are made and the model performances are investigated both with and without 
adjustments (oversampling) to see how models handle the class imbalance, and which benefit the most from attempts to 
correct for the imbalance.
Highly performed classification methods can benefit policy makers and planners to anticipate the energy demand and
infrastructure in California.

### Data Source
[California Vehicle Survey](https://www.energy.ca.gov/data-reports/surveys/california-vehicle-survey)

### Conclusions
No model appears to significantly outperform all others by all standards of model evaluation. However the diversity
of strengths and weaknesses across models suggests the potential for future application of machine learning methods
for vehicle ownership prediction.

### Notebooks
1. DataExplore - data exploration.
2. Performance - performance evaluation of different resampling techniques for each model. Similar evaluation for different Neural Net architectures.
3. Maps - true and prediction results visualized with CA maps, discussion about results and policy implementations.

### Code files
1. data_prepro.py - preprocessing and cleaning the three datasets, and merging into one.
2. data_plot.py - plotting functions for data exploration.
3. data_split.py - split dataset into train and test X and y, and resample.
4. model_logit.py - cross-validate hyperparameter, fit, predict, output performance variables, and model with best hyperparameters for Logistic Regression.
5. model_rf.py - cross-validate hyperparameter, fit, predict, output performance variables, and model with best hyperparameters for Random Forests.
6. model_knn.py - cross-validate hyperparameter, fit, predict, output performance variables, and model with best hyperparameters for K Nearest Neighbors.
7. model_nn.py - define and train model, predict, output accuracy, confusion matrix, roc, pr, and training arrays Neural Nets.
8. model_plot.py - plotting functions for model performance.
9. map_plot.py - plotting functions for mapping true and predicted results.

### Collaborators
* Koral Buch kbuch@ucdavis.edu
* Kenneth Broadhead kcbroadhead@ucdavis.edu

### References
*
*
*
*
