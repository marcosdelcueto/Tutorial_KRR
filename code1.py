#!/usr/bin/env python3
# Marcos del Cueto
# Import libraries
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

list_x = []
list_y = []
list_x_pred = []
random.seed(19)
for i in range(-5,6):
    x = i
    rnd_number= 2*random.uniform(-1,1)
    y = x + rnd_number
    list_x.append(x)
    list_y.append(y)
    print(x,y)

for i in range(-50,51):
    x = 0.1*i
    list_x_pred.append(x)

list_x = np.array(list_x).reshape(-1, 1)
list_x_pred = np.array(list_x_pred).reshape(-1, 1)

regr = linear_model.LinearRegression()
regr.fit(list_x,list_y)
list_y_pred = regr.predict(list_x_pred)

plt.plot(list_x_pred,list_y_pred,color='C1',linestyle='solid',linewidth=2)


print('Mean squared error: %.2f' % mean_squared_error(list_x_pred, list_y_pred))
print('Coefficient of determination: %.2f' % r2_score(list_x_pred, list_y_pred))


plt.scatter(list_x, list_y,color='C0')
file_name='Figure_1.png'
plt.savefig(file_name,format='png',dpi=600)
plt.close()
