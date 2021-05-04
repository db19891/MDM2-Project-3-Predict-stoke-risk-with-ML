# Import scikit-learn dataset library
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
dat = pd.read_csv(open("ReducedData1.csv"), header=None, na_values='nan')

#print(dat)


data = dat.values

imputer = KNNImputer()
imputer.fit(data)
Xtrans = imputer.transform(data)

print(Xtrans)

#for i in range(Xtrans.shape[1]):
   # count number of rows with missing values
#    n_miss = Xtrans[[i]].isnull().sum()
#   perc = n_miss / Xtrans.shape[0] * 100
#    print('> %d, Missing: %d (%.1f%%')

#print(data)

pd.DataFrame(Xtrans).to_csv('processed1.csv')
