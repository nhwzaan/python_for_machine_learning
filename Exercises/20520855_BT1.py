import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#read data
dataset = pd.read_csv('./Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, :-1].values
#print(X)
#print(y)

#split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

#print('LR innit')

#1. linear regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)


#print(y_pred)
#print(y_test)

#visuallization
plt.scatter(X_train, y_train, color='blue')
plt.plot(X_test, y_pred, color='red')
plt.show()

#2. decision tree
from sklearn import tree
clf = tree.DecisionTreeRegressor()
clf.fit(X_train, y_train)

DT_pred = clf.predict(X_test)


X_grid = np.arange(min(X_train), max(X_train), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)

plt.scatter(X, y, color='blue')
plt.plot(X_grid, DT_pred, color='red')
plt.show()


#3. SVR
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

from sklearn.svm import SVR

regressor = SVR(kernel='rbf')
regressor.fit(X,y)

y_pred = regressor.predict(6.5)

plt.scatter(X, y, color = 'magenta')
plt.plot(X, regressor.predict(X), color = 'green')
plt.show()

#4. Polynomial kernel
from sklearn.svm import SVC
svclassifier = SVC(kernel='poly', degree=8)
svclassifier.fit(X_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


#5. Random Forest
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)

regressor.fit(x, y)  

Y_pred = regressor.predict(np.array([6.5]).reshape(1, 1))

X_grid = np.arrange(min(x), max(x), 0.01)                
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x, y, color = 'blue')  
plt.plot(X_grid, regressor.predict(X_grid), color = 'green') 

plt.show()