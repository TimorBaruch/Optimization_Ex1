import numpy as np
import matplotlib.pyplot as plt

def plot_contour(f, x_lim, y_lim, title, paths_dict=None, levels_num=100):
    x_values = np.linspace(x_lim[0], x_lim[1])
    y_values = np.linspace(y_lim[0], y_lim[1])

    X, Y = np.meshgrid(x_values, y_values)
    Z = np.vectorize(lambda x1, x2: f(np.array([x1, x2]), False)[0])(X, Y)

    fig, ax = plt.subplots()
    contour = ax.contourf(X, Y, Z, levels_num, cmap='plasma')
    fig.colorbar(contour)
    ax.set_title(title)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')

    if paths_dict:
        for name, path in paths_dict.items():
            ax.plot(path[:, 0], path[:, 1], label=name)
        ax.legend()

    plt.show()

def plot_function_values(values_dict, title):
    fig, ax = plt.subplots()
    for label, values in values_dict.items():
        iterations = np.arange(len(values))
        ax.plot(iterations, values, marker='.', label=label)

    ax.set_title(title)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Function values')
    plt.legend()
    plt.show()