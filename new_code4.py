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

# Initialize lists and set random seed
list_x = []
list_y = []
list_x_pred = []
list_y_real = []
random.seed(2020)
# Create database with 21 points following quasi-lienar relation in interval x:[-5,5]
for i in range(-10,11):
    x = i/2
    rnd_number= random.uniform(-1,1)
    y = (x+4)*(x+1)*(x-1)*(x-3) + rnd_number
    list_x.append(x)
    list_y.append(y)
    print(x,y)
# Create list with 1060 points in interval x:[-5,5]
for i in range(-50,56):
    x = 0.1*i
    list_x_pred.append(x)
    list_y_real.append((x+4)*(x+1)*(x-1)*(x-3))
# Transform lists to np arrays
list_x = np.array(list_x).reshape(-1, 1)
list_x_pred = np.array(list_x_pred).reshape(-1, 1)
# Do linear regression using database with 21 points
list_y_pred = []
short_list_y_pred = []
rmse_list = []
# For each of the tested polynomial degree values
for alpha_value in [0.0001,0.1,10000,100000000]:
    krr = KernelRidge(alpha=alpha_value,kernel='polynomial',degree=4)
    krr.fit(list_x,list_y)
    list_y_pred.append(krr.predict(list_x_pred))
    new_y = krr.predict(list_x)
    short_list_y_pred.append(new_y)
    # Print rmse value
    rmse = math.sqrt(mean_squared_error(new_y, list_y))
    rmse_list.append(rmse)
    print('Root Mean Squared Error: %.2f' % rmse)
# Set axes and labels
fig, axs = plt.subplots(2, 2)
for ax in axs.flat:
    ax.set(xlabel='x', ylabel='y')
# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
# Subplot top-left
axs[0, 0].scatter(list_x, list_y,color='C0')
axs[0, 0].plot(list_x_pred,list_y_pred[0],color='C1')
axs[0, 0].set_title(r'$\alpha = 10^{-4}$')
axs[0, 0].set_xlim(-5.1,5.1)
axs[0, 0].set_ylim(-200,500)
axs[0, 0].annotate(u'$RMSE$ = %.1f' % rmse_list[0], xy=(0.15,0.85), xycoords='axes fraction')
# Subplot top-right
axs[0, 1].scatter(list_x, list_y,color='C0')
axs[0, 1].plot(list_x_pred,list_y_pred[1], color='C1')
axs[0, 1].set_title(r'$\alpha = 10^{-1}$')
axs[0, 1].set_xlim(-5.1,5.1)
axs[0, 1].set_ylim(-200,500)
axs[0, 1].annotate(u'$RMSE$ = %.1f' % rmse_list[1], xy=(0.15,0.85), xycoords='axes fraction')
# Subplot bottom-left
axs[1, 0].scatter(list_x, list_y,color='C0')
axs[1, 0].plot(list_x_pred,list_y_pred[2], color='C1')
axs[1, 0].set_title(r'$\alpha = 10^{4}$')
axs[1, 0].set_xlim(-5.1,5.1)
axs[1, 0].set_ylim(-200,500)
axs[1, 0].annotate(u'$RMSE$ = %.1f' % rmse_list[2], xy=(0.15,0.85), xycoords='axes fraction')
# Subplot bottom-right
axs[1, 1].scatter(list_x, list_y,color='C0')
axs[1, 1].plot(list_x_pred,list_y_pred[3], color='C1')
axs[1, 1].set_title(r'$\alpha = 10^{8}$')
axs[1, 1].set_xlim(-5.1,5.1)
axs[1, 1].set_ylim(-200,500)
axs[1, 1].annotate(u'$RMSE$ = %.1f' % rmse_list[3], xy=(0.15,0.85), xycoords='axes fraction')
# Print plot to file
file_name='Figure_4.png'
plt.savefig(file_name,format='png',dpi=600)
plt.close()
