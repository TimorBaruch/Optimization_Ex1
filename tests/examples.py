import numpy as np

def quadratic_example_1(x, is_hessian=False):
    f = np.dot(x, np.dot(np.array([[1, 0], [0, 1]]), x))
    g = 2 * np.dot(np.array([[1, 0], [0, 1]]), x)
    h = np.array([[2, 0], [0, 2]]) if is_hessian else None
    return f, g, h

def quadratic_example_2(x, is_hessian=False):
    f = np.dot(x, np.dot(np.array([[1, 0], [0, 100]]), x))
    g = 2 * np.dot(np.array([[1, 0], [0, 100]]), x)
    h = np.array([[2, 0], [0, 200]]) if is_hessian else None
    return f, g, h 

def quadratic_example_3(x, is_hessian=False):
    Q1 = np.array([[np.sqrt(3) / 2, -0.5], [0.5, np.sqrt(3) / 2]])
    Q2 = np.array([[100, 0], [0, 1]])
    Q = Q1.T.dot(Q2).dot(Q1)

    f = x.T.dot(Q).dot(x)
    g = 2 * Q.dot(x)
    h = 2 * Q if is_hessian else None
    return f, g, h

def rosenbrock_function(x, is_hessian=False):
    f = 100 * ((x[1] - (x[0] ** 2)) ** 2) + ((1 - x[0]) ** 2)
    g = np.array([-400 * x[0] * x[1] + 400 * (x[0] ** 3) + 2 * x[0] - 2,
                  200 * x[1] - 200 * (x[0] ** 2)])

    if is_hessian:
        h = np.array([[-400 * x[1] + 1200 * (x[0] ** 2) + 2, -400 * x[0]],
                      [-400 * x[0], 200]])
    else:
        h = None
    return f, g, h

def linear_function(x, is_hessian=False, a=[5, 5]):
    a = np.array(a)
    f = a.T.dot(x)
    g = a

    h = np.zeros((2, 2)) if is_hessian else None
    return f, g, h

def smoothed_triangle_function(x, is_hessian=False):
    f = np.exp(x[0] + 3 * x[1] - 0.1) + np.exp(x[0] - 3 * x[1] - 0.1) + np.exp(-x[0] - 0.1)
    g = np.array([np.exp(x[0] + 3 * x[1] - 0.1) + np.exp(x[0] - 3 * x[1] - 0.1) - np.exp(-x[0] - 0.1),
                  3 * np.exp(x[0] + 3 * x[1] - 0.1) - 3 * np.exp(x[0] - 3 * x[1] - 0.1)])

    if is_hessian:
        h = np.array([[np.exp(x[0] + 3 * x[1] - 0.1) + np.exp(x[0] - 3 * x[1] - 0.1) + np.exp(-x[0] - 0.1),
                       3 * np.exp(x[0] + 3 * x[1] - 0.1) - 3 * np.exp(x[0] - 3 * x[1] - 0.1)],
                      [3 * np.exp(x[0] + 3 * x[1] - 0.1) - 3 * np.exp(x[0] - 3 * x[1] - 0.1),
                       9 * np.exp(x[0] + 3 * x[1] - 0.1) + 9 * np.exp(x[0] - 3 * x[1] - 0.1)]])
    else:
        h = None
    return f, g, h
