import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import random

dataframe = read_csv('data\stroke_all_sampled.csv')     # Normalised and balanced data set with 7 features
train_all = dataframe.values

score = 0
recall_score = 0
spec_score = 0
precise_score = 0
pos = 0
neg = 0
pred_pos = 0
confidence = 0
true_confidence = 0
false_confidence = 0
no_loops = 100

model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

model.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

for i in range(no_loops):
    print(i,':')
    r = random.randrange(len(train_all))
    test = train_all[r,:]
    train = np.delete(train_all,r,0)
    test_features = [test[0:7]]
    test_features = np.asarray(test_features)
    test_labels = [test[7]]
    test_labels = np.asarray(test_labels)
    train_features = train[:,0:7]
    train_labels = train[:,7]

    #np.random.shuffle(train_labels)     # for testing random labeling


    feature_names = {'gender':0,'age':1,'hypertension':2,'heart disease':3,'avg glucose level':4,'bmi':5,'smokes':6}
    chosen_features = [1,4,5]
    test_features = test_features[:,chosen_features]
    train_features = train_features[:,chosen_features]

    train_features = np.where(train_features==float('nan'), 0, train_features)
    test_features = np.where(test_features==float('nan'), 0, test_features)


    test_labels = test_labels.astype(int)
    train_labels = train_labels.astype(int)
    

    model.fit(train_features, train_labels, epochs=50, batch_size=32, verbose=0)#, validation_data=(test_features,test_labels))
    test_loss, test_acc = model.evaluate(test_features, test_labels, verbose=0)
    print('\nTest accuracy:', test_acc)
    pred_confidence = model.predict(test_features)
    predictions = np.round(pred_confidence[:,1])

    if test_acc == 1:
        score+=1
        if predictions[:].astype(int) == 0:
            true_confidence = true_confidence + pred_confidence[:,0]
        else:
            true_confidence = true_confidence + pred_confidence[:,1]
    else:
        if predictions[:].astype(int) == 0:
            false_confidence = false_confidence + pred_confidence[:,0]
        else:
            false_confidence = false_confidence + pred_confidence[:,1]
    
    if predictions[:].astype(int) == 0:
        confidence = confidence + pred_confidence[:,0]
    else:
        confidence = confidence + pred_confidence[:,1]

    print('prediction: ', predictions[:].astype(int), '    true: ', test_labels[:])

    if test_labels == 1:
        pos += 1
        if predictions == 1:
            recall_score += 1

    if predictions == 1:
        pred_pos += 1
        if test_labels == 1:
            precise_score += 1

    elif predictions == 0:
        neg += 1
        if test_labels == 0:
            spec_score += 1


print('\nscore = ', round((score/no_loops)*100,2), '%')
print('\nrecall = ', round((recall_score/pos)*100,2), '%')
print('\nspecifictity = ', round((spec_score/neg)*100,2), '%')
print('\nprecision = ', round((precise_score/pred_pos)*100,2), '%')
print('\navg confidence = ', np.round((confidence[:]/no_loops)*100,2), '%')
print('\navg confidence (when correct) = ', np.round((true_confidence[:]/score)*100,2), '%')
print('\navg confidence (when wrong) = ', np.round((false_confidence[:]/(no_loops-score))*100,2), '%')
print('fin')