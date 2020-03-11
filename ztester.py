import numpy as np


A_list = [
    [1, 2, 3, 4],
    [2, 4, 6, 8],
    [3, 6, 9, 12],
    [4, 8, 12, 16],
    [5, 10, 15, 20],
    [6, 12, 18, 24]
]

X_list = [[1], [1], [1], [1]]

if A_list is not np.ndarray:
    A_array = np.asarray(A_list)
else:
    A_array = A_list

if X_list is not np.ndarray:
    X_array = np.asarray(X_list)
else:
    X_array = X_list

B = np.matmul(A_array, X_array)
print(B)
