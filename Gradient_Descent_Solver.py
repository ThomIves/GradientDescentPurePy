from Plot_Tools import Basic_Plot as BP
import random


class Gradient_Descent_Solver:
    def __init__(self, X, Y, LR,
                 ci=1000, tol=1e-12,
                 max_cnt=1e9, rnd=6):
        self.X = X
        self.Y = Y
        self.LR = LR
        self.ci = ci
        self.tol = tol
        self.max_cnt = max_cnt
        self.rnd = rnd

        self.num_records = len(X)
        self.num_dims = len(X[0])

        self.Yp = [0] * self.num_records
        self.delta = [0] * self.num_records
        self.randomize_weights()

        self.cnt_list = []
        self.cost_list = []

    def set_weights(self, ws):
        self.ws = ws

    def set_labels(self, Y):
        self.Y = Y

    def randomize_weights(self):
        self.ws = [random.random()] * self.num_dims

    def model(self, X):
        num_records = len(X)
        self.Yp = [0] * num_records

        for i in range(num_records):
            for j in range(self.num_dims):
                self.Yp[i] += X[i][j] * self.ws[j]

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
        for i in range(self.num_records):
            self.delta[i] = self.Yp[i] - self.Y[i]
            for j in range(self.num_dims):
                self.ws[j] = \
                    self.ws[j] - self.LR * self.X[i][j] * self.delta[i]

    def __cost__(self):
        total_cost = 0
        for value in self.delta:
            total_cost += value ** 2

        return total_cost ** 0.5

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
        ws = [round(x, 6) for x in self.ws]
        print(f'Solved Weights: {ws}')
        print(f'Iteration Steps to Solution: {self.count}')

    def plot_solution_convergence(self):
        BP(self.cnt_list, self.cost_list,
           t='Cost vs. Solution Steps',
           x_t='Solution Steps', y_t='Cost')

    def plot_predictions(self, X, Y, col_of_X=1):
        Xsp = [row[col_of_X] for row in self.X]
        Xtp = [row[col_of_X] for row in X]
        BP(Xtp, self.Yp, xp=Xsp, yp=self.Y,
           t='Model Predictions vs. Inputs',
           x_t='Inputs',
           y_t='Predictions and Original Output')
