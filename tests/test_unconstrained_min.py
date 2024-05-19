import unittest
from src.utils import *
from src.unconstrained_min import *
from tests.examples import *

class TestUnconstrainedMinimization(unittest.TestCase):
    def run_all_tests(self, name, f, x0=[1.0, 1.0], obj_tol=10**-12, param_tol=10**-8, max_iter=100,
                contour_x_lim=[-2, 2], contour_y_lim=[-2, 2], contour_levels=100):

        optimizer = LineSearchMinimizer()
        x0 = np.array(x0) if isinstance(x0, list) else x0
        grad_x, grad_f, grad_success, grad_history = optimizer._minimize(f, x0.copy(), obj_tol, param_tol, max_iter, 'gradient_descent')
        newton_x, newton_f, newton_success, newton_history = optimizer._minimize(f, x0.copy(), obj_tol, param_tol, max_iter, 'newton')

        # Save the paths and the values to dict format for visualization
        # paths = {
        #     'Gradient Descent': np.array(grad_history['path']),
        #     'Newton': np.array(newton_history['path']),
        # }
        #
        # values = {
        #     'Gradient Descent': grad_history['values'],
        #     'Newton': newton_history['values']
        # }

        # Print the final results
        print("<----- Summary of the Results ----->")
        print(f'{name:} - {len(grad_history["path"])-1} Iterations, Final location: ({grad_x[0]},{grad_x[1]}), Final value: {grad_f}, Minimum: {grad_success}' )
        print(f'{name:} - {len(newton_history["path"])-1} Iterations, Final location: ({newton_x[0]},{newton_x[1]}), Final value: {newton_f}, Minimum: {newton_success}' )
        print()

        # Visualize the results
        plot_function_values({'Gradient Descent': grad_history['values'], 'Newton': newton_history['values']},
                             title=f'Function values vs Iterations of {name}')
        plot_contour(f, contour_x_lim, contour_y_lim, paths_dict={'Gradient Descent': np.array(grad_history['path']),
                            'Newton': np.array(newton_history['path'])}, levels_num=contour_levels, title=f'The Contour of {name} Function')

    def test_quadratic_examples(self):
        print('--------------------------START---------------------------')
        self.run_all_tests("Quadratic Example 1", quadratic_example_1)
        print('-----------------------------------------------------')
        self.run_all_tests("Quadratic Example 2", quadratic_example_2)
        print('-----------------------------------------------------')
        self.run_all_tests("Quadratic Example 3", quadratic_example_3)
        print('-----------------------------------------------------')

    def test_rosenbrock(self):
        self.run_all_tests("Rosenbrock", rosenbrock_function, x0=[-1.0, 2.0], contour_y_lim=[-2, 5], max_iter=10000)
        print('-----------------------------------------------------')

    def test_linear_function(self):
        self.run_all_tests("Linear Function", linear_function, contour_x_lim=[-500, 2], contour_y_lim=[-500, 2])
        print('-----------------------------------------------------')

    def test_smoothed_triangle_function(self):
        self.run_all_tests("Smoothed Triangle Function", smoothed_triangle_function, contour_x_lim=[-1, 1], contour_y_lim=[-1, 1], contour_levels=50)
        print('--------------------------END---------------------------')


# Run the tests
# unittest.main(argv=[''], exit=False)
# if __name__ == '__main__':
#     unittest.main()
if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit:
        pass
