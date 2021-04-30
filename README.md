# MDM2-Project-3-SVM Overview


This repository contains a mixture of python files - used to program the Support vector machines machine learning experimentation used in MDM2 Project 3 on Stroke prediction. Along with csv files of the unprocessed and processed dataset on stroke prediction. The dataset used can be found @ https://www.kaggle.com/fedesoriano/stroke-prediction-dataset.

For all python scripts contained, "Final1.csv" is the processed dataset used to train, test and predict with the classifier, apart from the shuffletest python script which predictably uses "shuffleddata.csv" in place. The pre-processing of the dataset has been completed in MATLAB, therefore unfortunately there is no record of this contained within this repository.

# Python Scripts
"linear_svm" trains a classifier using a linear kernel with the leave-one-out cross validation method and returning test accuaracy.

"mkl_test" uses the "linear_svm" program as a base, but replaces the linear kernel with polynomial, rbf and sigmoid kernels, comparing the test accuracy of each kernel with the linear kernel.

"gradient_boost_acc" trains on the dataset using a gradient boosting classifier instead of kernels. The method continues to use the leave-one-out cross validation method and compute a test accuracy.

"shuffletest" is an identical program to "gradient_boost_acc" apart from the dataset that is used. This program is a test of validity of our stroke predictions.

"gradient_boost_graphical" builds on "gradient_boost_acc" by producing a calibration plot that visualises the accuracy of the predictions made by the classifier. Alongside the gradient boosting method there is a calibration plot for the logistic regression classifier. This aims to give a confidence level to our predictions. Another visualisation seen here is a comparison of which features are the most important on stroke prediction.
