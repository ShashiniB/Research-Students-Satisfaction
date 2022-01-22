import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

np.random.seed(70)
#read CSV file
dataset= pd.read_csv("New3.csv")

print(dataset.shape)

x = dataset.iloc[:,  12:23]
y = dataset.iloc[:, 23]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=400)

print(x_train.shape, x_test.shape)

# replace missing values
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp = imp.fit(x_train)

x_train_imp = imp.transform(x_train)
x_test_imp = imp.transform(x_test)

# min maxScaler
from sklearn.preprocessing import MinMaxScaler
data = MinMaxScaler()
normalized_data = data.fit(x_train)

x_train_scaled = normalized_data.transform(x_train)
x_test_scaled = normalized_data.transform(x_test)

print(x_train_scaled)


from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(5,5,5), activation='relu', solver='adam', max_iter=1000)
mlp.fit(x_train, y_train)

x_predict = mlp.predict(x_train)
y_predict = mlp.predict(x_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(y_predict)

# performance of the model on training data
print(confusion_matrix(y_train, x_predict))
print(classification_report(y_train, x_predict))

# performance of the model on the test
print(confusion_matrix(y_test, y_predict))
print(classification_report(y_test, y_predict))

print("Accuracy is", accuracy_score(y_test, y_predict)*100)