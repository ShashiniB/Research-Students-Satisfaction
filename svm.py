import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
from sklearn.impute import SimpleImputer

np.random.seed(70)
# loading data file
dataset = pd.read_csv("New3.csv")

print(dataset.head())
print(dataset.shape)

x = dataset.iloc[:, 12:23]
y = dataset.iloc[:, 23]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=80)


# train

model = SVC(kernel='linear')
model.fit(x_train, y_train)


y_predict = model.predict(x_test)
print(y_predict)

accuracy = model.score(x_test, y_test)

print(accuracy)
print("accuracy", metrics.accuracy_score(y_test, y_predict)*100)
print(metrics.classification_report(y_test, y_predict))


