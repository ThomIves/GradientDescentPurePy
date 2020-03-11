import Gradient_Descent_Solver_with_Numpy as GDSnp
import numpy as np
import sys

# #############################################################################
# Setup
it = [x/10 for x in list(range(10))]
X = [[x**0, x**1, x**2] for x in it]
wa = [4.0, 0.1, 0.2]  # [[4.0], [0.1], [0.2]] with numpy


def Y_maker(X, w):
    Y = []
    for x in X:
        y = 0
        for i in range(len(w)):
            y += x[i] * wa[i]
        Y.append(y)

    return Y


Y = Y_maker(X, wa)
Y = [[y] for y in Y]
LR = 0.001

# Cheat to get Y easier
gds = GDSnp.Gradient_Descent_Solver_with_Numpy(X, Y, LR)

# Solve
gds.train()

# Report
gds.report_results()
gds.plot_solution_convergence()

# Test
it = [x/100 for x in list(range(100))]
Xt = [[x**0, x**1, x**2] for x in it]

Yt = gds.model(Xt)

# Report
gds.plot_predictions(Xt, Yt, col_of_X=1)
