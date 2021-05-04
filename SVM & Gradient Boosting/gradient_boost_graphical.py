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

""" BALANCED AND IMPUTED FILE """

Data = pd.read_csv("Final1.csv")

# Give the data a set of axis as the initial data has lost axis in the pre-processing
Data = Data.set_axis(["1","2","3","4","5","6","7","8","9","10","11"],axis=1)

feat_names = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']

a = len(Data) # An average accuracy over the number of iterations, a, is needed after the cross-validation.

Acc = 0

for i in range(1,a):

   Test = Data.iloc[[i]]
   Train = Data.drop([i])
   
   TsL = Test.drop(["1","2","3","4","5","6","7","8","9","10"],axis=1)
   TrL = Train.drop(["1","2","3","4","5","6","7","8","9","10"],axis=1)
   TsF = Test.drop(["11"],axis=1)
   TrF = Train.drop(["11"],axis=1)

   TsL = TsL.values.ravel()  # rewrites into correct format
   TrL = TrL.values.ravel()


   clf = svm.SVC(kernel='linear') # Linear Kernel
   clf.fit(TrF, TrL)

   y_pred = clf.predict(TsF)
   Acc += metrics.accuracy_score(TsL, y_pred)
   #print(Acc)

Accuracy = Acc/a
#print(Accuracy)

target = Data["11"]
del Data["11"]
TrF, TsF, TrL, TsL = train_test_split(Data, target, test_size=0.3, random_state=509) # 70% training and 30% test

# Create classifiers
lr = LogisticRegression()
gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)

# Plot calibration plots

plt.figure(figsize=(10, 10))
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)

ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
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

    ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
            label="%s" % (name, ))

    # compute the test accuracy and brier score accuracy for the classifier
    y_pred = clf.predict(TsF)
    clf_score = brier_score_loss(TsL, prob_pos, pos_label=target.max())
    print("%s:" % name)
    print("\tAccuracy:",metrics.accuracy_score(TsL, y_pred))
    print("\tBrier(p): %1.3f" % (clf_score))
    #print("\tPrecision: %1.3f" % precision_score(TsL, y_pred))
    #print("\tRecall: %1.3f" % recall_score(TsL, y_pred))
    #print("\tF1: %1.3f\n" % f1_score(TsL, y_pred))

ax1.set_xlabel("Mean predicted value")
ax1.set_ylabel("Fraction of positives")
ax1.set_ylim([-0.05, 1.05])
ax1.legend(loc="lower right")
ax1.set_title('Calibration plots  (reliability curve)')

plt.tight_layout()
plt.show()

# Use Gradient Boosting to find the most important features in a dataset

params = {'n_estimators': 500,
          'max_depth': 4,
          'min_samples_split': 5,
          'learning_rate': 0.01,
          'loss': 'ls'}

reg = ensemble.GradientBoostingRegressor(**params)
reg.fit(TrF, TrL) # train the classifier on the data

mse = mean_squared_error(TsL, reg.predict(TsF))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))

test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
for i, y_pred in enumerate(reg.staged_predict(TsF)):
    test_score[i] = reg.loss_(TsL, y_pred) # optimise the differentiable loss

feature_importance = reg.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
fig = plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, np.array(feat_names)[sorted_idx])
plt.title('Feature Importance (MDI)')

result = permutation_importance(reg, TsF, TsL, n_repeats=10,
                                random_state=42, n_jobs=2)
sorted_idx = result.importances_mean.argsort()
plt.subplot(1, 2, 2)
plt.boxplot(result.importances[sorted_idx].T,
            vert=False, labels=np.array(feat_names)[sorted_idx])
plt.title("Permutation Importance (test set)")
fig.tight_layout()
plt.show()