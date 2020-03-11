import Gradient_Descent_Solver_with_Numpy as GDSnp
import numpy as np
import random

# #############################################################################
# Section 1: Create Fake X Data
it = [x/100 for x in list(range(100))]
X = [[x**0, x**1, x**2] for x in it]
wa = [4.0, 0.1, 0.2]


# Section 2: Create Fake Y Data
def Y_maker(X, w):
    Y = []
    for x in X:
        y = 0
        for i in range(len(w)):
            y += x[i] * wa[i]
        Y.append(y + y / 100 * random.random())

    return Y


Y = Y_maker(X, wa)
Y = [[y] for y in Y]

# # Section 3: Instantiate Our Class
LR = 0.001
gds = GDSnp.Gradient_Descent_Solver_with_Numpy(X, Y, LR)

# Section 4: Train model and Show Results
gds.train()
gds.report_results()
gds.plot_solution_convergence()

# Section 5: Create Fake Test Inputs, Predict, Plot
it = [x/100 for x in list(range(100))]
Xt = [[x**0, x**1, x**2] for x in it]
Yt = gds.model(Xt)
gds.plot_predictions(Xt, Yt, col_of_X=1)
