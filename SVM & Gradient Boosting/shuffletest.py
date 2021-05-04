from sklearn import datasets, svm, metrics, pipeline, preprocessing, impute, inspection, ensemble, calibration, linear_model, naive_bayes
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import shogun as sg
from shogun import *
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, precision_score, recall_score, f1_score, mean_squared_error)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance

Data = pd.read_csv("shuffleddata.csv") # Note the dataset is a pre-processed shuffled dataset

# Give the data a set of axis as the initial data has lost axis in the pre-processing
Data = Data.set_axis(["1","2","3","4","5","6","7","8","9","10","11"],axis=1)

feat_names = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']

# Create classifiers
lr = LogisticRegression()
gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)

a = len(Data) # An average accuracy over the number of iterations, a, is needed after the cross-validation.

Acc = 0
brier = 0

for i in range(1,a):

   Test = Data.iloc[[i]] # Take the test set as a single item
   Train = Data.drop([i])
   
   # Splitting the data into labels and features
   # The labels, stroke or no stroke, is in column 11, the other columns after the features.
   TsL = Test.drop(["1","2","3","4","5","6","7","8","9","10"],axis=1)
   TrL = Train.drop(["1","2","3","4","5","6","7","8","9","10"],axis=1)
   TsF = Test.drop(["11"],axis=1)
   TrF = Train.drop(["11"],axis=1)

   TsL = TsL.values.ravel()  # rewrites into correct format
   TrL = TrL.values.ravel()

   for clf, name in [(lr, 'Logistic'),
                     (gbc, 'Gradient Boosting Classifier')]:
      clf.fit(TrF, TrL) # train with the classifier
      # Predict with the classifier
      if hasattr(clf, "predict_proba"):
         prob_pos = clf.predict_proba(TsF)[:, 1]
      else:  # use decision function
         prob_pos = clf.decision_function(TsF)
         prob_pos = \
               (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
      fraction_of_positives, mean_predicted_value = \
         calibration_curve(TsL, prob_pos, n_bins=10)

      # compute the test accuracy and brier score accuracy for the classifier
      y_pred = clf.predict(TsF)
      clf_score = brier_score_loss(TsL, prob_pos, pos_label=TsL.max())
      print("%s:" % name)
      print("\tAccuracy:",metrics.accuracy_score(TsL, y_pred))
      print("\tBrier(p): %1.3f" % (clf_score))
      #print("\tPrecision: %1.3f" % precision_score(TsL, y_pred))
      #print("\tRecall: %1.3f" % recall_score(TsL, y_pred))
      #print("\tF1: %1.3f\n" % f1_score(TsL, y_pred))

      if name=='Gradient Boosting Classifier': # cumulate the accuracy and brier for the gradient boosting classifier for cross-validation method
         Acc += metrics.accuracy_score(TsL, y_pred)
         brier += clf_score
      else:
         Acc = Acc
         brier = brier

Accuracy = Acc/a
print("Accuracy:", Accuracy)
Brierscore = brier/a
print("Brier Score:", Brierscore)




















