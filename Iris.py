# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 13:33:36 2019

@author: sidha
"""
## Importing The Libraries
import pandas as pd
import matplotlib.pyplot as plt

## Importing a File
dataset = pd.read_csv("Iris.csv")
X = dataset.iloc[: , 1:5].values 
Y = dataset.iloc[:, -1].values

## Encoding the Categorical Variable
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

## Visulaising
## Parallel COordinates
from pandas.plotting import parallel_coordinates
plt.figure(figsize = (15, 10))
parallel_coordinates(dataset.drop("Id", axis = 1),"Species")
plt.title("Parallel Coordinates Plot", fontsize = 20, fontweight = "bold")
plt.xlabel("Features", fontsize = 15)
plt.ylabel("Features Values", fontsize = 15)
plt.legend(loc = 1, prop = {"size" : 15}, frameon = True,shadow = True, facecolor = "White",
           edgecolor = "black")
plt.show()

## Andrew Curves
from pandas.plotting import andrews_curves
andrews_curves(dataset.drop("Id", axis = 1),"Species")
plt.title("Andrews Curve Plot",fontsize = 20, fontweight = "bold")
plt.xlabel("Features", fontsize = 15)
plt.ylabel("Features Values", fontsize = 15)
plt.legend(loc =1, prop = {"size" : 5}, frameon = True, shadow = True, facecolor = "White",
           edgecolor = "black")
plt.show()

## BoxPlots
plt.figure()
dataset.drop("Id", axis = 1).boxplot(by = "Species", figsize = (15,10))
plt.show()

## KNN Algorithm
## Splitting the dataset into Training and Test Set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

## Fitting Classifier to the Training Set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p =2)
classifier.fit(X_train, Y_train)

## Predicting the Test set Results
Y_pred = classifier.predict(X_test)

## Making a Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

from sklearn.metrics import accuracy_score
r = accuracy_score(Y_test,Y_pred)
print(r*100)


