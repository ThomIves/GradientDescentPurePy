from Plot_Tools import Basic_Plot as BP
import random
import sys


def model(X, w):
    records = len(X)
    Yp = [0] * records
    dims = len(X[0])

    dim_chk = sum([dims == len(xi) for xi in X]) == records
    assert dim_chk, 'NOT ALL X RECORDS HAVE SAME DIMENSIONS'

    for i in range(records):
        for j in range(dims):
            Yp[i] += X[i][j] * w[j]

    return Yp


def delta_calc(Yp, Y):
    records = len(X)
    delta = [0] * records

    for i in range(records):
        delta[i] = Yp[i] - Y[i]

    return delta


def update_weights(w, delta, X, LR):
    records = len(X)
    dims = len(X[0])

    for i in range(records):
        for j in range(dims):
            w[j] = w[j] - LR * X[i][j] * delta[i]

    return w


def cost(delta):
    total_cost = 0
    for value in delta:
        total_cost += value ** 2

    return total_cost


def record_value(count, comm_interval):
    if count % comm_interval == 0:
        return True
    else:
        return False


def solver(X, w, Y, LR, ci,
           tol=1.0e-16, max_cnt=1e9):

    cost_delta = 1.0
    cost_last = 1.0
    cnt = 0
    cost_list = []
    cnt_list = []
    w_list = []

    while cost_delta > tol:
        Yp = model(X, w)
        delta = delta_calc(Yp, Y)
        w = update_weights(w, delta, X, LR)

        cost_now = cost(delta)
        cost_delta = abs(cost_last - cost_now)
        cost_last = cost_now

        if record_value(cnt, ci):
            cost_list.append(cost_now)
            cnt_list.append(cnt)
            w_list.append(w.copy())

        cnt += 1

        if cnt > max_cnt:
            print("Exiting Due To Exceeding Max Iterations")
            break

    return w, cost_list, delta, cnt, cnt_list, w_list


# #############################################################################
# Setup
# Section 1: Fake Training Data Preparation
Xp = [2, 2, 4, 4]
X = [[1, x] for x in Xp]
Y = [2.1, 1.9, 3.1, 2.9]
ws = [random.random()] * len(X[0])
LR = 0.01

# Section 2: Solve / train
ws, cost_list, delta, cnt, cnt_list, w_list = \
    solver(X, ws, Y, LR, 1)

# Section 3: Report training results
print(f'Delta = {delta}, Count = {cnt}')
ws = [round(x, 6) for x in ws]
print(f'Solved Weights: {ws}')

BP(cnt_list, cost_list,
   t='cost vs. Step', x_t='Step', y_t='cost')

# Plot learning path of weights
for i in range(len(ws)):
    wi_list = [pair[i] for pair in w_list]
    title = f'Cost vs Weight {i}'
    x_t = f'Weight {i}'
    y_t = 'Cost'
    BP(wi_list, cost_list,
       t=title, x_t=x_t, y_t=y_t)

# Section 4: Fake test data creation and predictions
Xtp = [x/20.0 for x in list(range(100))]
Xtc = [[1, x] for x in Xtp]
YtP = model(Xtc, ws)

BP(Xtp, YtP, xp=Xp, yp=Y,
   t='Predictions vs. Fake X',
   x_t='Fake X', y_t='Predicted Y')
