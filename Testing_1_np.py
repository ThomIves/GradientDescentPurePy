import Gradient_Descent_Solver_with_Numpy as GDSnp
import numpy as np
import sys

# #############################################################################
# Setup
X = [[1, 2],
     [1, 4]]
LR = 0.001

# Cheat to get Y easier
Y = [[0], [0]]
wa = [[1.0], [0.5]]
gds = GDSnp.Gradient_Descent_Solver_with_Numpy(X, Y, LR)
gds.set_weights(wa)
Y = gds.model(X)

# Now that we have Y andomize weights and set Y = labels
gds.randomize_weights()
gds.set_labels(Y)

# Solve
gds.train()

# Report
gds.report_results()
gds.plot_solution_convergence()

# Test
Xt = [[1, 0],
      [1, 3],
      [1, 5]]
Yt = gds.model(Xt)

# Report
gds.plot_predictions(Xt, Yt, col_of_X=1)
