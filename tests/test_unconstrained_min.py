import unittest
from src.utils import *
from src.unconstrained_min import *
from tests.examples import *

class TestUnconstrainedMinimization(unittest.TestCase):
    def run_all_tests(self, name, f, x0=[1.0, 1.0], obj_tol=10**-12, param_tol=10**-8, max_iter=100,
                contour_x=[-2, 2], contour_y=[-2, 2], contour_levels=100):

        optimizer = LineSearchMinimizer()
        x0 = np.array(x0) if isinstance(x0, list) else x0
        grad_x, grad_f, grad_success, grad_history = optimizer._minimize(f, x0.copy(), obj_tol, param_tol, max_iter, 'gradient_descent')
        newton_x, newton_f, newton_success, newton_history = optimizer._minimize(f, x0.copy(), obj_tol, param_tol, max_iter, 'newton')

        # Print the final results to console
        print()
        print("<----- The Final Iteration Results ----->")
        print(f'{name}:')
        print(f'{"Method":<20} {"Iterations":<15} {"Final Location":<26} {"Final Value":<15} {"Minimum Found":<15}')
        print('-' * 100)

        grad_location = f'({grad_x[0]:.4g}, {grad_x[1]:.4g})'
        newton_location = f'({newton_x[0]:.4g}, {newton_x[1]:.4g})'

        print(
            f'{"Gradient Descent":<20} {len(grad_history["path"]) - 1:<15} {grad_location:<26} {grad_f:<15.6g} {str(grad_success):<15}')
        print(
            f'{"Newton":<20} {len(newton_history["path"]) - 1:<15} {newton_location:<26} {newton_f:<15.6g} {str(newton_success):<15}')
        print()

        # Visualize the results
        plot_function_values({'Gradient Descent': grad_history['values'], 'Newton': newton_history['values']},
                             title=f'Function values vs Iterations of {name}')
        plot_contour(f, contour_x, contour_y, paths_dict={'Gradient Descent': np.array(grad_history['path']),
                            'Newton': np.array(newton_history['path'])}, levels_num=contour_levels, title=f'The Contour and algorithm paths of {name}')

    def test_quadratic_examples(self):
        print('--------------------------START---------------------------')
        self.run_all_tests("Quadratic Example 1", quadratic_example_1)
        print('-----------------------------------------------------')
        self.run_all_tests("Quadratic Example 2", quadratic_example_2)
        print('-----------------------------------------------------')
        self.run_all_tests("Quadratic Example 3", quadratic_example_3)
        print('----------------------------------------------------------')

    def test_rosenbrock(self):
        self.run_all_tests("Rosenbrock", rosenbrock_function, x0=[-1.0, 2.0], contour_y=[-2, 5], max_iter=10000)
        print('-----------------------------------------------------')

    def test_linear_function(self):
        self.run_all_tests("Linear Function", linear_function, contour_x=[-500, 2], contour_y=[-500, 2])
        print('-----------------------------------------------------')

    def test_smoothed_triangle_function(self):
        self.run_all_tests("Smoothed Triangle Function", smoothed_triangle_function, contour_x=[-1, 1], contour_y=[-1, 1], contour_levels=20)
        print('--------------------------END---------------------------')


# Run the tests
if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit:
        pass
