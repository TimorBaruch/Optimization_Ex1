import numpy as np

class LineSearchMinimizer:
    def __init__(self):
        self.path_history = {'path': [], 'values': []}
    
    def _minimize(self, f, x0, obj_tol, param_tol, max_iter, method): 
        self.path_history = dict(path=[], values=[])

        if method == 'gradient_descent':
            return self.gradient_descent(f, x0, obj_tol, param_tol, max_iter)
        elif method == 'newton':
            return self.newton(f, x0, obj_tol, param_tol, max_iter)
        else:
            raise ValueError("Invalid method. Use 'gradient_descent' or 'newton'.")

    def gradient_descent(self, f, x0, obj_tol, param_tol, max_iter):
        is_min_found = False
        curr_val, curr_grad, _ = f(x0, False)

        x = x0
        print(f'----- Gradient Descent -----')
        print(f'Iteration {"0":>4}/{max_iter} - location={x}\tobj_value={curr_val}')
        self.path_history['path'].append(x.copy())
        self.path_history['values'].append(curr_val)

        for i in range(max_iter):
            direction = -curr_grad
            step_len = self.wolfe_condition_with_backtracking(f, x, curr_val, curr_grad, direction)

            # Update x
            prev_x = x.copy()
            x += step_len * direction

            # Update value and gradient
            prev_val, prev_grad = curr_val, curr_grad
            curr_val, curr_grad, _ = f(x, False)

            # Print the current values
            print(f'Iteration {i+1:>4}/{max_iter} - location={x}\tobj_value={curr_val}')
            self.path_history['path'].append(x.copy())
            self.path_history['values'].append(curr_val)

            # Check if reached min or converged
            if abs(curr_val - prev_val) < obj_tol or np.linalg.norm(x - prev_x) < param_tol or not curr_grad.any():
                is_min_found = True
                break

        return x, curr_val, is_min_found, self.path_history

    def newton(self, f, x0, obj_tol, param_tol, max_iter):
        is_min_found = False
        curr_val, curr_grad, curr_hess = f(x0, True)

        x = x0
        print(f'----- Newton -----')
        print(f'Iteration {"0":>4}/{max_iter} - location={x}\tobj_value={curr_val}')
        self.path_history['path'].append(x.copy())
        self.path_history['values'].append(curr_val)

        for i in range(max_iter):
            try:
                direction = np.linalg.solve(curr_hess, -curr_grad)
            except np.linalg.LinAlgError:
                # If the Hessian is not positive definite
                break

            step_len = self.wolfe_condition_with_backtracking(f, x, curr_val, curr_grad, direction)

            # Update x
            prev_x = x.copy()
            x += step_len * direction

            # Update value, gradient, and Hessian
            prev_val, prev_grad, prev_hess = curr_val, curr_grad, curr_hess
            curr_val, curr_grad, curr_hess = f(x, True)

            # Print the current values
            print(f'Iteration {i+1:>4}/{max_iter} - location={x}\tobj_value={curr_val}')
            self.path_history['path'].append(x.copy())
            self.path_history['values'].append(curr_val)

            # Check if reached min or converged
            if abs(curr_val - prev_val) < obj_tol or np.linalg.norm(x - prev_x) < param_tol or not curr_grad.any():
                is_min_found = True
                break

        return x, curr_val, is_min_found, self.path_history

    
    def wolfe_condition_with_backtracking(self, f, x, val, gradient, direction, condition_const=0.01, backtrack_const=0.5):
        step_size = 1.0
        curr_val, _, _ = f(x + step_size * direction, False)

        while curr_val > val + condition_const * step_size * gradient.dot(direction):
            step_size *= backtrack_const
            curr_val, _, _ = f(x + step_size * direction, False)

        return step_size

