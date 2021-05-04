""" PYTHON FILE THAT EXECUTES AND FINDS THE ACCUARACY WITH THE 4 MAIN KERNELS """

from sklearn import datasets
import pandas as pd
from sklearn import svm
from sklearn import metrics
import numpy as np

Data = pd.read_csv("Final1.csv")

# Give the data a set of axis as the initial data has lost axis in the pre-processing
Data = Data.set_axis(["1","2","3","4","5","6","7","8","9","10","11"],axis=1)

def kerneltype(k,ispoly,Data):

   a = len(Data) # An average accuracy over the number of iterations, a, is needed after the cross-validation.

   Acc = 0

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

      # classify the support vector machine
      if ispoly:
         clf = svm.SVC(kernel=k,degree=2) # polynomial kernel requires an extra definition
      else:
         clf = svm.SVC(kernel=k)

      clf.fit(TrF, TrL) # train the classifier

      y_pred = clf.predict(TsF) # predict on the test features
      Acc += metrics.accuracy_score(TsL, y_pred) # accuracy measured against the test labels
      #print(Acc)

   Accuracy = Acc/a
   print(k,":")
   print(Accuracy)

kerneltype('linear',False,Data)
kerneltype('rbf',False,Data)
kerneltype('poly',True,Data)
kerneltype('sigmoid',False,Data)




















