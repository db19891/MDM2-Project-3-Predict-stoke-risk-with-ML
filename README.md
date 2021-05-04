# Overview


This repository contains a mixture of python files - used to program the Support vector machines machine and Neural Network methods used in MDM2 Project 3 on Stroke prediction. Also contained is the code used to pre-processd data along with csv files of the unprocessed and processed dataset on stroke prediction. The dataset used can be found @ https://www.kaggle.com/fedesoriano/stroke-prediction-dataset.

# Pre-process data

This directory contains the code used in the pre-processing of the data. This code uses MATLAB for the most part since MATLAB is better suited to manipulating datasets. The scripts balance the number of stroke and no stroke labels as well as changing string data into integer. The python scikit-learn library has a kNNimpute tool to predict missing values, which are prevalent in our dataset, and a python script that imputes for the missing values is found here also. The final processed dataset is "Final1.csv" and found in the "SVM & Gradient Boosting" directory.

# SVM & Gradient Boosting

For all python scripts contained, "Final1.csv" is the processed dataset used to train, test and predict with the classifier, apart from the shuffletest python script which predictably uses "shuffleddata.csv" in place.

"linear_svm" trains a classifier using a linear kernel with the leave-one-out cross validation method and returning test accuaracy.

"mkl_test" uses the "linear_svm" program as a base, but replaces the linear kernel with polynomial, rbf and sigmoid kernels, comparing the test accuracy of each kernel with the linear kernel.

"gradient_boost_acc" trains on the dataset using a gradient boosting classifier instead of kernels. The method continues to use the leave-one-out cross validation method and compute a test accuracy.

"shuffletest" is an identical program to "gradient_boost_acc" apart from the dataset that is used. This program is a test of validity of our stroke predictions.

"gradient_boost_graphical" builds on "gradient_boost_acc" by producing a calibration plot that visualises the accuracy of the predictions made by the classifier. Alongside the gradient boosting method there is a calibration plot for the logistic regression classifier. This aims to give a confidence level to our predictions. Another visualisation seen here is a comparison of which features are the most important on stroke prediction.

# Neural Networks

A python file script is used to code a neural network to predict stroke risk. The script utilises the open-source library TensorFlow, which has comprehensive tools designed for Machine Learning and works especially well with neural networks. The processed dataset used here is alongside - It has been pre-processed in the same way as elsewhere in the repository.

# References

Scikit learn gradient boosting - https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html

Scikit Learn calibration plots - https://scikit-learn.org/stable/auto_examples/calibration/plot_compare_calibration.html#sphx-glr-auto-examples-calibration-plot-compare-calibration-py

Scikit-learn kNNImputer - https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html

TensorFlow toolkit - https://www.tensorflow.org/
