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
        val_curr, grad_curr, _ = f(x0, False)
        x = x0

        self.path_history['path'].append(x.copy())
        self.path_history['values'].append(val_curr)

        print(f'*---------- Gradient Descent Method ----------*')
        print(f'Iteration {"0":>5}/{max_iter} =\tlocation={x}\tobj_value={val_curr}')

        for i in range(max_iter):
            direction = -grad_curr
            step_len = self.wolfe_condition_with_backtracking(f, x, val_curr, grad_curr, direction)

            # Update the x
            x_prev = x.copy()
            x += step_len * direction

            # Update function value and gradient
            val_prev, grad_prev = val_curr, grad_curr
            val_curr, grad_curr, _ = f(x, False)

            # Save results
            self.path_history['path'].append(x.copy())
            self.path_history['values'].append(val_curr)

            # Print the current values
            print(f'Iteration {i+1:>5}/{max_iter} =\tlocation={x}\tobj_value={val_curr}')

            # Stopping criteria
            obj_diff = abs(val_curr - val_prev)
            param_diff = np.linalg.norm(x - x_prev)

            if obj_diff < obj_tol or param_diff < param_tol or not grad_curr.any():
                is_min_found = True
                break

        return x, val_curr, is_min_found, self.path_history

    def newton(self, f, x0, obj_tol, param_tol, max_iter):
        is_min_found = False
        val_curr, grad_curr, hes_curr = f(x0, True)
        x = x0

        self.path_history['path'].append(x.copy())
        self.path_history['values'].append(val_curr)

        print(f'*---------- Newton Method ----------*')
        print(f'Iteration {"0":>5}/{max_iter} =\tlocation={x}\tobj_value={val_curr}')

        for i in range(max_iter):
            if hes_curr is None:
                break
            try:
                direction = np.linalg.solve(hes_curr, -grad_curr)
            except np.linalg.LinAlgError:
                break

            step_len = self.wolfe_condition_with_backtracking(f, x, val_curr, grad_curr, direction)

            # Update the x
            x_prev = x.copy()
            x += step_len * direction

            # Update value, gradient and Hessian
            val_prev, grad_prev, hes_prev = val_curr, grad_curr, hes_curr
            val_curr, grad_curr, hes_curr = f(x, True)

            # Save results
            self.path_history['path'].append(x.copy())
            self.path_history['values'].append(val_curr)

            # Print the current values
            print(f'Iteration {i+1:>5}/{max_iter} =\tlocation={x}\tobj_value={val_curr}')

            # Stopping criteria
            obj_diff = abs(val_curr - val_prev)
            param_diff = np.linalg.norm(x - x_prev)

            if obj_diff < obj_tol or param_diff < param_tol or not grad_curr.any():
                is_min_found = True
                break

        return x, val_curr, is_min_found, self.path_history
    
    def wolfe_condition_with_backtracking(self, f, x, val, grad, direction, condition_const=0.01, backtrack_const=0.5):
        step_size = 1.0
        val_curr = f(x + step_size * direction, False)[0]

        # First wolfe condition
        while val_curr > val + condition_const * step_size * grad.dot(direction):
            step_size *= backtrack_const
            val_curr = f(x + step_size * direction, False)[0]

        return step_size
