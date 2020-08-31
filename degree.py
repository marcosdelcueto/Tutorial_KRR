#!/usr/bin/env python3
# Marcos del Cueto
# Import libraries
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.kernel_ridge import KernelRidge

list_x = []
list_y = []
list_x_pred = []
list_y_real = []
random.seed(2020)
for i in range(-10,12):
    x = i/2
    rnd_number= 1*random.uniform(-1,1)
    #y = x + rnd_number
    y = (x+4)*(x+1)*(x-1)*(x-3) + rnd_number
    #y = (x+4)*(x+1)*(x-1)*(x-3)
    list_x.append(x)
    list_y.append(y)
    print(x,y)

for i in range(-50,60):
    x = 0.1*i
    list_x_pred.append(x)
    list_y_real.append((x+4)*(x+1)*(x-1)*(x-3))

list_x = np.array(list_x).reshape(-1, 1)
list_x_pred = np.array(list_x_pred).reshape(-1, 1)

#regr = linear_model.LinearRegression()
#regr.fit(list_x,list_y)
#list_y_pred = regr.predict(list_x_pred)

list_y_pred = []
short_list_y_pred = []
for degree_value in [2,3,4,5]:
#for alpha_value in [0.0001,0.001,0.1,1]:
    krr = KernelRidge(alpha=1.0,kernel='polynomial',degree=degree_value)
    #krr = KernelRidge(alpha=alpha_value,kernel='rbf')
    krr.fit(list_x,list_y)
    list_y_pred.append(krr.predict(list_x_pred))
    new_y = krr.predict(list_x)
    short_list_y_pred.append(new_y)
    #print('list_y:', list_y)
    #print('new_y:', new_y)
    print('Root Mean Squared Error: %.2f' % math.sqrt(mean_squared_error(new_y,list_y)))
    #print('Coefficient of determination: %.2f' % r2_score(new_y,list_y))




fig, axs = plt.subplots(2, 2)
axs[0, 0].scatter(list_x, list_y,color='C0')
axs[0, 0].plot(list_x_pred,list_y_pred[0],color='C1')
axs[0, 0].set_title(r'$d = 2$')
axs[0, 0].set_xlim(-5.1,5.1)
axs[0, 0].set_ylim(-200,500)
axs[0, 1].scatter(list_x, list_y,color='C0')
axs[0, 1].plot(list_x_pred,list_y_pred[1], color='C1')
axs[0, 1].set_title(r'$d = 3$')
axs[0, 1].set_xlim(-5.1,5.1)
axs[0, 1].set_ylim(-200,500)
axs[1, 0].scatter(list_x, list_y,color='C0')
axs[1, 0].plot(list_x_pred,list_y_pred[2], color='C1')
axs[1, 0].set_title(r'$d = 4$')
axs[1, 0].set_xlim(-5.1,5.1)
axs[1, 0].set_ylim(-200,500)
axs[1, 1].scatter(list_x, list_y,color='C0')
axs[1, 1].plot(list_x_pred,list_y_pred[3], color='C1')
axs[1, 1].set_title(r'$d = 5$')
axs[1, 1].set_xlim(-5.1,5.1)
axs[1, 1].set_ylim(-200,500)

for ax in axs.flat:
    ax.set(xlabel='x', ylabel='y')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

#plt.plot(list_x_pred,list_y_pred,color='C1',linestyle='solid',linewidth=2)
#print('Mean squared error: %.2f' % mean_squared_error(list_y, list_y_pred))
#print('Coefficient of determination: %.2f' % r2_score(list_y, list_y_pred))
#plt.scatter(list_x, list_y,color='C0')

file_name='degree.png'
plt.savefig(file_name,format='png',dpi=600)
plt.close()
