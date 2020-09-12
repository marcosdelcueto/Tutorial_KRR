#!/usr/bin/env python3
# Marcos del Cueto
# Import libraries
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Initialize lists and set random seed
list_x = []
list_y = []
list_x_pred = []
random.seed(19)
# Create database with 11 points following quasi-lienar relation in interval x:[-5,5]
for i in range(-5,6):
    x = i
    rnd_number = random.uniform(-2,2)
    y = x + rnd_number
    list_x.append(x)
    list_y.append(y)
    print(x,y)
# Create list with 1010 points in interval x:[-5,5]
for i in range(-50,51):
    x = 0.1*i
    list_x_pred.append(x)
# Transform lists to np arrays
list_x = np.array(list_x).reshape(-1, 1)
list_x_pred = np.array(list_x_pred).reshape(-1, 1)
# Do linear regression using database with 11 points
regr = linear_model.LinearRegression()
regr.fit(list_x,list_y)
# Calculate value of linear regressor at 101 points in interval x:[-5,5]
list_y_pred = regr.predict(list_x_pred)
# Calculate value of linear regressor at 11 points in interval x:[-5,5]
new_y = regr.predict(list_x)
# Print rmse value
rmse = math.sqrt(mean_squared_error(new_y, list_y))
print('Root Mean Squared Error: %.2f' % rmse)
# Set axes and labels
fig = plt.figure()
ax = fig.add_subplot()
ax.set_xlim(-5.5,5.5)
ax.set_ylim(-5.5,5.5)
ax.xaxis.set_ticks(range(-5,6))
ax.yaxis.set_ticks(range(-5,6))
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.annotate(u'$RMSE$ = %.1f' % rmse, xy=(0.15,0.85), xycoords='axes fraction')
# Plot as orange line the regression line at interval
plt.plot(list_x_pred,list_y_pred,color='C1',linestyle='solid',linewidth=2)
# Plot as blue points the original database
plt.scatter(list_x, list_y,color='C0')
# Print plot to file
file_name='Figure_1.png'
plt.savefig(file_name,format='png',dpi=600)
plt.close()
