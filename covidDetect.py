import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import pickle
from dataprep.eda.missing import plot_missing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix
warnings.filterwarnings("ignore")
import math

covid = pd.read_csv('CovidDataset.csv')
covid.replace('Yes', 1, inplace=True)
covid.replace('No', 0, inplace=True)
x = covid.drop('COVID-19', axis=1)
y = covid['COVID-19']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)
clf = RandomForestClassifier(max_depth=None, random_state=0)
clf.fit(x_train, y_train)

pickle.dump(clf, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))