import Gradient_Descent_Solver as GDS

# #############################################################################
# Setup
X = [[1, 2, 4],
     [1, 4, 16],
     [1, 6, 36]]
LR = 0.001

# Cheat to get Y easier
Y = [5, 17, 37]
gds = GDS.Gradient_Descent_Solver(X, Y, LR)

# Solve
gds.train()

# Report
gds.report_results()
gds.plot_solution_convergence()

# Test
Xt = [[1, 0, 0],
      [1, 2, 4],
      [1, 3, 9],
      [1, 4, 16],
      [1, 5, 25],
      [1, 6, 36],
      [1, 7, 49]]

Yt = gds.model(Xt)

# Report
gds.plot_predictions(Xt, Yt, col_of_X=1)
