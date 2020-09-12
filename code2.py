#!/usr/bin/env python3
# Marcos del Cueto
# Import libraries
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

list_x = []
list_y = []
list_x_pred = []
random.seed(2020)
for i in range(-10,12):
    x = i/2
    rnd_number= random.uniform(-1,1)
    #y = x + rnd_number
    y = (x+4)*(x+1)*(x-1)*(x-3) + rnd_number
    list_x.append(x)
    list_y.append(y)
    print(x,y)

for i in range(-50,56):
    x = 0.1*i
    list_x_pred.append(x)

list_x = np.array(list_x).reshape(-1, 1)
list_x_pred = np.array(list_x_pred).reshape(-1, 1)

regr = linear_model.LinearRegression()
regr.fit(list_x,list_y)
list_y_pred = regr.predict(list_x_pred)
new_y = regr.predict(list_x)

plt.plot(list_x_pred,list_y_pred,color='C1',linestyle='solid',linewidth=2)


print('Root Mean Squared Error: %.2f' % math.sqrt(mean_squared_error(new_y,list_y)))


plt.scatter(list_x, list_y,color='C0')
file_name='Figure_2.png'
plt.savefig(file_name,format='png',dpi=600)
plt.close()
