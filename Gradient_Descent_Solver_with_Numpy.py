from Plot_Tools import Basic_Plot as BP
import numpy as np
import sys


class Gradient_Descent_Solver_with_Numpy:
    def __init__(self, X, Y, LR,
                 ci=1000, tol=1e-12,
                 max_cnt=1e9, rnd=6):

        if type(X) is not np.ndarray:
            X = np.asarray(X)
        if type(Y) is not np.ndarray:
            Y = np.asarray(Y)

        self.X = X
        self.Y = Y
        self.LR = LR
        self.ci = ci
        self.tol = tol
        self.max_cnt = max_cnt
        self.rnd = rnd

        self.num_records = len(X)
        self.num_dims = len(X[0])

        self.Yp = np.zeros((self.num_records, 1))
        self.delta = np.zeros((self.num_records, 1))
        self.randomize_weights()

        self.cnt_list = []
        self.cost_list = []

    def set_weights(self, ws):
        if type(ws) is not np.ndarray:
            ws = np.asarray(ws)

        self.ws = ws

    def set_labels(self, Y):
        if Y is not np.ndarray:
            Y = np.asarray(Y)

        self.Y = Y

    def randomize_weights(self):
        self.ws = np.random.rand(self.num_dims, 1)

    def model(self, X):
        if X is not np.ndarray:
            X = np.asarray(X)
        self.Yp = np.matmul(X, self.ws)
        return self.Yp

    def train(self):
        cost_delta = 1.0
        cost_last = 1.0

        self.count = 0
        self.cost_list = []
        self.cnt_list = []

        while cost_delta > self.tol and self.__iterations_below_max__():
            self.model(self.X)
            self.__update_weights__()

            self.cost_now = self.__cost__()
            cost_delta = abs(cost_last - self.cost_now)
            cost_last = self.cost_now

            self.__record_values__()

    def __update_weights__(self):
        self.delta = self.Yp - self.Y
        dcdw = np.sum(self.X * self.delta, axis=0).reshape((self.num_dims, 1))
        self.ws = self.ws - self.LR * dcdw

    def __cost__(self):
        return np.sum(np.square(self.delta)) ** 0.5

    def __record_values__(self):
        self.count += 1

        if self.count % self.ci == 0:
            self.cost_list.append(self.cost_now)
            self.cnt_list.append(self.count)

    def __iterations_below_max__(self):
        if self.count < self.max_cnt:
            return True
        else:
            print("Exceeded Max Iterations")
            return False

    def report_results(self):
        ws = np.around(self.ws, decimals=6)
        print(f'Solved Weights: {ws}')
        print(f'Iteration Steps to Solution: {self.count}')

    def plot_solution_convergence(self):
        BP(self.cnt_list, self.cost_list,
           t='Cost vs. Solution Steps',
           x_t='Solution Steps', y_t='Cost')

    def plot_predictions(self, X, Y, col_of_X=1):
        if type(X) is not np.ndarray:
            X = np.asarray(X)
        Xsp = self.X[:, col_of_X].tolist()
        Xtp = X[:, col_of_X].tolist()

        BP(Xtp, self.Yp, xp=Xsp, yp=self.Y,
           t='Model Predictions vs. Inputs',
           x_t='Inputs',
           y_t='Predictions and Original Output')
