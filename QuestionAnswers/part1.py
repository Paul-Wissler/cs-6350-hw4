import numpy as np


def q5():
    print('PART 1 QUESTION 5')
    x1 = np.array([[0.5], [-1], [1.5]])
    x2 = np.array([[-1], [-2], [0.2]])
    x3 = np.array([[0.3], [-2], [-2.5]])
    y = np.array([[1], [-1], [1]])
    w = np.array([0, 0, 0])
    b = 0
    X = np.hstack([x1, x2, x3])
    C = 1/3
    gamma = [0.01, 0.005, 0.0025]
    print('X matrix:\n', X)
    print('y vector:\n', y)
    print('hyper parameter C:', C)
    print('initial weight vector:\n', w)
    print('initial bias:', b)

    for i in range(3):
        gamma_i = gamma[i]
        print(f'\nGamma for iteration {i + 1}: {gamma_i}')
        pass_check = compute_check_case(X.T[i], w, y[i])
        print('Case is <= 1?', pass_check)
        if pass_check:
            print('Update according to case 1')
            w, b = case_1_update(w, gamma_i, b, C, len(X[i]), y[i], X.T[i])
        else:
            print('Update according to case 2')
            w = case_2_update(w, gamma_t)
        print(f'Iteration {i + 1} update:')
        print('New weight vector:\n', w)
        print('New bias:', b)

    # RESULTS:
    # PART 1 QUESTION 5
    # TEST 1.0
    # X matrix:
    #  [[ 0.5 -1.   0.3]
    #  [-1.  -2.  -2. ]
    #  [ 1.5  0.2 -2.5]]
    # y vector:
    #  [[ 1]
    #  [-1]
    #  [ 1]]
    # hyper parameter C: 0.3333333333333333
    # initial weight vector:
    #  [0 0 0]
    # initial bias: 0

    # Gamma for iteration 1: 0.01
    # Case is <= 1? True
    # Update according to case 1
    # Iteration 1 update:
    # New weight vector:
    #  [ 0.005 -0.01   0.015]
    # New bias: 0.009999999999999998

    # Gamma for iteration 2: 0.005
    # Case is <= 1? True
    # Update according to case 1
    # Iteration 2 update:
    # New weight vector:
    #  [9.9750e-03 5.0000e-05 1.3925e-02]
    # New bias: 0.004999999999999999

    # Gamma for iteration 3: 0.0025
    # Case is <= 1? True
    # Update according to case 1
    # Iteration 3 update:
    # New weight vector:
    #  [ 0.01070006 -0.00495012  0.00764019]
    # New bias: 0.007499999999999999



def compute_check_case(x_i: np.ndarray, w: np.ndarray, y_i: int) -> bool:
    return (y_i * w.T * x_i <= 1)[0]
 

def case_1_update(w: np.ndarray, gamma_t: float, b: float, C: float, N: int, y_i, x_i):
    return (w - gamma_t * w + gamma_t * C * N * y_i * x_i), (b + gamma_t * C * N * y_i * 1)[0]


def case_2_update(w: np.ndarray, gamma_t: float):
    return (1 - gamma_t) * w
