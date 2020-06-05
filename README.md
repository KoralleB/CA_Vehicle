# Vehicle Ownership - Powertrain Prediction

### Abstract
In this project, our objective is to identify the demographic and socioeconomic attributes of alternative fuel vehicle owners California. For this, we use the National Household Travel Survey (NHTS) 2017 dataset. We investigate several different machine learning methods that can handle unbalanced data, such as logistic regression, support vector machine, random forest, neural nets, and others. The prediction performance will be assessed with the most important variables for each model. Conclusions about the affecting attributes will be made on a county resolution.

Since less than 10% of the vehicles in California are alternative fuel vehicles, we will treat the unbalanced dataset accordingly, but it might impose difficulties on the analysis. The dataset includes about 25,000 households and we hope that is big enough to achieve good prediction performance.

### Data Source
[California Vehicle Survey](https://www.energy.ca.gov/data-reports/surveys/california-vehicle-survey)

### Models to be considered
* Logistic regression
* Random forests
* KNN
* Nnets

### Notebooks
1. DataExplore - data exploration.
2. Resample - performance evaluation of different resampling techniques for each model.
3. Performance - performance evaluation of the different models with the best resampling technique.
4. Maps - true and prediction results visualized with CA maps, discussion about results and policy implementations.

### Code files
1. prepro.py - preprocessing and cleaning the three datasets, and merging into one.
2. data_plot.py - plotting functions for data exploration.
3. split.py - split dataset into train and test X and y, and resample.
4. model_logit.py - cross-validate hyperparameter, fit, predict, output performance variables, and model with best hyperparameters for logistic regression.
5. model_rf.py - cross-validate hyperparameter, fit, predict, output performance variables, and model with best hyperparameters for random forests.
6. model_plot.py - plotting functions for model performance.
7. map_plot.py - plotting functions for mapping true and predicted results.

### Collaborators
* Koral Buch kbuch@ucdavis.edu
* Kenneth Broadhead kcbroadhead@ucdavis.edu
