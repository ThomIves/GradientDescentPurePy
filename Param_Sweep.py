from Plot_Tools import Basic_Plot as BP
import random
import sys

# Section 1: Fake Training Data Preparation
Xp = [4]
X = [[1, x] for x in Xp]
Y = 3

# Section 2: Solution of Model Parameters / Weights
b = 1.0  # Fixed value for y-intercept - testing
m = 0.0  # initial value for slope
LR = 0.0001  # smaller learning rate for illustration

cost_list = []
m_list = []

num_pts = len(X)
for it in range(10000):
    Yp = X[0][0] * b + X[0][1] * m
    delta = Yp - Y
    cost = delta ** 2.0
    dm = X[0][1] * delta
    m = m - LR * dm
    if it % 50 == 0:
        cost_list.append(cost)
        m_list.append(m)  # b_list.append(m)

# SPECIAL SECTION: Parameter sweep plot of m vs. cost
sweep_m_list = []
sweep_cost_list = []

# sweep of slope vs. cost
for val in range(0, 10001):
    slope = val / 10000.0
    cost = ((b + X[0][1] * slope) - Y) ** 2.0

    sweep_m_list.append(slope)
    sweep_cost_list.append(cost)

BP(x=sweep_m_list, y=sweep_cost_list,
   xp=m_list, yp=cost_list,
   t='Cost vs. Values of Slope - Dots Show Solution Steps',
   x_t='Slope (m) Values', y_t='Cost Function Values')
