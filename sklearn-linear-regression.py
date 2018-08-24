import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

plt.rcParams['figure.figsize'] = (20.0, 10.0)

# Reading data
data = pd.read_csv('headbrain.csv')
print(data.shape)
data.head()

# Collecting X and Y
X = data['Head Size(cm^3)'].values
Y = data['Brain Weight(grams)'].values

# Mean X and Y
mean_x = np.mean(X);
mean_y = np.mean(Y);

# Total number of values
m = len(X)

# Using thenformula to calculate b1 and b2
numer = 0
denom = 0

for i in range(m):
    numer += (X[i] - mean_x) * (Y[i] - mean_y)
    denom += (X[i] - mean_x) ** 2

b1 = numer / denom
b0 = mean_y - (b1 * mean_x)

# Print coefficients
print('Brainweight = ',b0,' + ',b1,' * HeadSize');

# Plotting values and regression line

max_x = np.max(X) + 100
min_x = np.min(X) - 100

# Calculating line values x and y
x = np.linspace(min_x, max_x, 1000)
y = b0 + b1 * x

# Plotting line
plt.plot(x, y, color='#58b970', label='Regression Line')

# Plotting scatter points
plt.scatter(X, Y, c='#ef5423', label='Scatter Plot')

plt.xlabel('Head Size in cm3')
plt.ylabel('Brain weight in grams')
plt.legend()
plt.show()

# Cannot use Rank 1 matrix in scikit learn
X = X.reshape((m,1))

# Creating model
reg = LinearRegression()

# Fitting training data
reg = reg.fit(X,Y)

# Y prediction
Y_pred = reg.predict(X)

# Calculating RMSE and R2 Score
mse = mean_squared_error(Y, Y_pred)
rmse = np.sqrt(mse)

r2_score = reg.score(X, Y)

print(np.sqrt(mse))
print(r2_score)

