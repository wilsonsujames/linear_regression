import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
#Ten baseline variables, age, sex, body mass index, average blood pressure, 
# and six blood serum measurements were obtained for each of n = 442 diabetes patients,
#  as well as the response of interest, a quantitative measure of disease progression one year after baseline.

#https://www4.stat.ncsu.edu/~boos/var.select/diabetes.tab.txt
#把正個資料集印出來，發現是一個tuple(array[[]],array[[]])的格式
print(datasets.load_diabetes(return_X_y=True))

diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

print("印出前兩筆病人的特徵數值:::")
#發現資料已經經過standlize處理，應該是用(x-mean)/標準差
print(diabetes_X[0])
print(diabetes_X[1])

print("***************")



# Use only one feature,BMI
#diabetes_X[:,2]為一維array，轉成2維numpy array model才吃得進去
diabetes_X = diabetes_X[:, 2, np.newaxis]


#最後20筆拿來測試
# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

#y Column 11 is a quantitative measure of disease progression one year after baseline 經過一年，糖尿病進展指數
# Split the targets into training/testing sets
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

# The coefficients
#迴歸係數
print('Coefficients: \n', regr.coef_)
# The mean squared error
#均方誤差
print('Mean squared error: %.2f'
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# The coefficient of determination: 1 is perfect prediction
#R square
print('Coefficient of determination: %.2f'
      % r2_score(diabetes_y_test, diabetes_y_pred))

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()

