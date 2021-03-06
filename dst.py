import classifier as classifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler


np.random.seed(75)
dataset = pd.read_csv("New3.csv")

print(dataset.head())

x = dataset.values[:, 12:23]
y = dataset.values[:, 23]
y = y.astype('int')

print(x.shape)
print(y.shape)

# split dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=85)

# replace missing values
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp = imp.fit(x_train)

x_train_imp = imp.transform(x_train)
x_test_imp = imp.transform(x_test)

# min max scaler
data = MinMaxScaler()

normalized_data = data.fit(x_train_imp)

x_train_scaled = normalized_data.transform(x_train_imp)
x_test_scaled = normalized_data.transform(x_test_imp)

print(x_train_scaled)

# train

model_DT = DecisionTreeClassifier()
model_DT.fit(x_train_scaled, y_train)

# Funcion to prediction
y_predict = model_DT.predict(x_test_scaled)

print(y_predict)

accuracy = model_DT.score(x_test_scaled, y_test)

print(accuracy)

print("accuracy", metrics.accuracy_score(y_test, y_predict)*100)
print(metrics.classification_report(y_test, y_predict))
