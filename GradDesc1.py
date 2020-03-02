from Plot_Tools import Basic_Plot as BP
import random
import sys

# Section 1: Fake Training Data Preparation
Xp = [2, 2, 4, 4]
X = [[1, x] for x in Xp]
Y = [2.1, 1.9, 3.1, 2.9]
BP(xp=Xp, yp=Y,
   t='Initial Training Points for X and Y')

# Section 2: Solution of Model Parameters / Weights
b = 0.01  # initial value for y-intercept
m = 0.025  # initial value for slope
Yp = [0, 0, 0, 0]
LR = 0.001

cost_list = []
b_list = []
m_list = []

num_pts = len(X)
for it in range(100000):
    for i in range(num_pts):
        # Prediction based on current weights
        Yp[i] = X[i][0] * b + X[i][1] * m

        # Error based on current weights
        delta = Yp[i] - Y[i]
        cost = delta ** 2.0

        # The gradients for y-intercept and slope
        db = X[i][0] * delta
        dm = X[i][1] * delta

        # Update the weights
        b = b - LR * db
        m = m - LR * dm

    if it % 100 == 0:
        cost_list.append(cost)
        b_list.append(b)
        m_list.append(m)

# Section 3: Show convergence of weights
BP(x=b_list, y=cost_list,
   t='Convergence of b Value vs. Cost',
   x_t='b Values', y_t='Cost',
   xp=b_list, yp=cost_list)

BP(m_list, cost_list,
   t='Convergence of m Value vs. Cost',
   x_t='m Values', y_t='Cost',
   xp=m_list, yp=cost_list)

# Section 4: Create Fake Test Data
Xtp = [x/20.0 for x in list(range(100))]
Xtc = [[1, x] for x in Xtp]
Yt = []

# Section 5: Predictions Using Fake Test Data
for x in Xtc:
    yt = b * x[0] + m * x[1]
    Yt.append(yt)

BP(x=Xtp, y=Yt,
   t='Y Predictions Using Trained Model',
   xp=Xp, yp=Y)
