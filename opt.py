import optuna
from methods_from_lab1 import *
from main import *

optuna.logging.set_verbosity(optuna.logging.WARNING)


def rosenbrock(x):
    return 100 * (x[1] - x[0]    ** 2) ** 2 + (1 - x[0]) ** 2


def himmelblau(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


def quadratic(x):
    return 1000 * x[0] ** 2 + x[1] ** 2


x0_range = [-10.0, 10.0]

test_functions = [
    ("Rosenbrock", rosenbrock),
    ("Himmelblau", himmelblau),
    ("Quadratic", quadratic)
]

methods_lab1 = [
    ("GD_constant", gradient_descent_with_constant_step, 1),
    ("GD_armijo", gradient_descent_armijo, 1),
    ("GD_wolfe", gradient_descent_wolfe, 1),
    ("GD_golden", gradient_descent_with_golden_section, 1),
    ("GD_dichotomy", gradient_descent_dichotomy, 1)
]

methods_lab2 = [
    ("Newton_wolfe", newton_method_wolfe_step, 2),
    ("Newton_golden", newton_method_golden_section, 2),
    ("BFGS", bfgs_method, 2)
]

best_params = {func_name: {} for func_name, _ in test_functions}

x, y = sp.symbols('x y')
rosenbrock_sympy = 100 * (y - x ** 2) ** 2 + (1 - x) ** 2
himmelblau_sympy = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
quadratic_sympy = 1000 * x ** 2 + y ** 2

test_functions_sympy = [
    ("Rosenbrock", rosenbrock, rosenbrock_sympy),
    ("Himmelblau", himmelblau, himmelblau_sympy),
    ("Quadratic", quadratic, quadratic_sympy)
]

for func_name, func in test_functions:
    print(f"\n=== Optimizing for {func_name} function ===")
    # if func_name == "Rosenbrock":
    #     x0 = np.array([-1.5, 1.5])
    # else:
    #     x0 = np.array([15.0, -15.0])
    for method_name, method, lab_number in methods_lab1:
        print(f"\nMethod: {method_name} from lab: {lab_number}")


        def objective(trial):
            # if func_name == "Rosenbrock" and (method_name == "GD_dichotomy" or method_name == "GD_golden"):
                # x0 = np.array([-5, 5])
            
            # else:
                # x0 = np.array([
                #     trial.suggest_float('x0_0', x0_range[0], x0_range[1]),
                #     trial.suggest_float('x0_1', x0_range[0], x0_range[1])
                # ])
            x0 = np.array([
                    trial.suggest_float('x0_0', x0_range[0], x0_range[1]),
                    trial.suggest_float('x0_1', x0_range[0], x0_range[1])
                ])    
            if method_name == "GD_constant":
                step = trial.suggest_float('step', 1e-6, 1.0, log=True)
                eps = trial.suggest_float('eps', 1e-8, 1e-4, log=True)
                max_iter = trial.suggest_int('max_iter', 100, 10000)
                x_min = method(func, x0.copy(), step=step, max_iter=max_iter, eps=eps)

            elif method_name == "GD_decreasing":
                init_step = trial.suggest_float('init_step', 0.1, 10.0)
                eps = trial.suggest_float('eps', 1e-8, 1e-4, log=True)
                max_iter = trial.suggest_int('max_iter', 100, 10000)
                x_min = method(func, x0.copy(), step=init_step, max_iter=max_iter, eps=eps)

            elif method_name == "GD_armijo":
                init_step = trial.suggest_float('init_step', 0.1, 10.0)
                eps = trial.suggest_float('eps', 1e-8, 1e-4, log=True)
                max_iter = trial.suggest_int('max_iter', 100, 10000)
                x_min = method(func, x0.copy(), step=init_step, max_iter=max_iter, eps=eps)

            elif method_name == "GD_wolfe":
                init_step = trial.suggest_float('init_step', 0.1, 10.0)
                eps = trial.suggest_float('eps', 1e-8, 1e-4, log=True)
                max_iter = trial.suggest_int('max_iter', 100, 10000)
                if func_name == "Rosenbrock":
                    c1 = trial.suggest_float('a', 0.0, 0.0)
                    c2 = trial.suggest_float('b', 0.1, 0.1)
                else:
                    c1 = trial.suggest_float('a', 0.0, 0.1)
                    c2 = trial.suggest_float('b', 0.1, 1.0)

                x_min = method(func, x0.copy(), step=init_step, c1=c1, c2=c2, max_iter=max_iter, eps=eps)

            elif method_name == "GD_golden":
                if func_name == "Rosenbrock":
                    a = trial.suggest_float('a', 0.0, 0.0)
                    b = trial.suggest_float('b', 0.1, 0.1)
                else:
                    a = trial.suggest_float('a', 0.0, 0.1)
                    b = trial.suggest_float('b', 0.1, 1.0)
                
                eps = trial.suggest_float('eps', 1e-8, 1e-4, log=True)
                max_iter = trial.suggest_int('max_iter', 100, 10000)
                x_min = method(func, x0.copy(), a=a, b=b, max_iter=max_iter, eps=eps)
                # x_min = method(func, x0.copy(), a=0, b=0.1, max_iter=max_iter, eps=eps)

            elif method_name == "GD_dichotomy":
                if func_name == "Rosenbrock":
                    a = trial.suggest_float('a', 0.0, 0.0)
                    b = trial.suggest_float('b', 0.1, 0.1)
                else:
                    a = trial.suggest_float('a', 0.0, 0.1)
                    b = trial.suggest_float('b', 0.1, 1.0)
                eps = trial.suggest_float('eps', 1e-8, 1e-4, log=True)
                max_iter = trial.suggest_int('max_iter', 100, 10000)
                x_min = method(func, x0.copy(), a=a, b=b, max_iter=max_iter, eps=eps)
                # x_min = method(func, x0.copy(), a=0, b=0.1, max_iter=max_iter, eps=eps)

            return func(x_min)


        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)

        best_params[func_name][method_name] = study.best_params
        print(f"Best params: {study.best_params}")
        print(f"Best value: {study.best_value}")


for func_name, func, sympy_func in test_functions_sympy:
    print(f"\n=== Optimizing advanced methods for {func_name} function ===")
    # if func_name == "Rosenbrock":
    #     x0 = np.array([-1.5, 1.5])
    # else:
    #     x0 = np.array([15.0, -15.0])

    for method_name, method, lab_number in methods_lab2:
        print(f"\nMethod: {method_name} from lab: {lab_number}")

        def objective(trial):
            x0 = np.array([
                trial.suggest_float('x0_0', x0_range[0], x0_range[1]),
                trial.suggest_float('x0_1', x0_range[0], x0_range[1])
            ])
            
            eps = trial.suggest_float('eps', 1e-8, 1e-4, log=True)

            if method_name == "Newton_wolfe":
                wolfe_a = trial.suggest_float('wolfe_a', 0.1, 10.0)
                wolfe_c1 = trial.suggest_float('wolfe_c1', 0.01, 0.2)
                wolfe_c2 = trial.suggest_float('wolfe_c2', 0.7, 0.99)
                max_iter = trial.suggest_int('max_iter', 100, 10000)
                x_min, _ = method(func, sympy_func, [x, y],
                                x0=x0.copy(),
                                max_iter=max_iter,
                                eps=eps,
                                wolfe_params={'a': wolfe_a, 'c1': wolfe_c1, 'c2': wolfe_c2})

            elif method_name == "Newton_golden":
                golden_a = trial.suggest_float('golden_a', 0.0, 0.1)
                golden_b = trial.suggest_float('golden_b', 0.1, 1.0)
                max_iter = trial.suggest_int('max_iter', 100, 10000)
                x_min, _ = method(func, sympy_func, [x, y],
                                x0=x0.copy(),
                                max_iter=max_iter,
                                eps=eps,
                                golden_params={'a': golden_a, 'b': golden_b})

            elif method_name == "BFGS":
                bfgs_a = trial.suggest_float('bfgs_a', 0.0, 0.1)
                bfgs_b = trial.suggest_float('bfgs_b', 0.1, 1.0)
                max_iter = trial.suggest_int('max_iter', 100, 10000)
                x_min, _ = method(func,
                                x0=x0.copy(),
                                max_iter=max_iter,
                                eps=eps,
                                golden_params={'a': bfgs_a, 'b': bfgs_b})

            return func(x_min)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)

        best_params[func_name][method_name] = study.best_params
        print(f"Best params: {study.best_params}")
        print(f"Best value: {study.best_value}")




print("\n=== Testing with optimized parameters ===")
# x0_test = np.array([-1.5, 1.5])
for func_name, func in test_functions:
    print(f"\nFunction: {func_name}")
    # if func_name == "Rosenbrock":
    #     x0_test = np.array([-1.5, 1.5])
    # else:
    #     x0_test = np.array([15.0, -15.0])
    for method_name, method, _ in methods_lab1:

        if method_name not in best_params[func_name]:
            continue

        params = best_params[func_name][method_name]
        print(f"\nMethod: {method_name}")
        # if func_name == "Rosenbrock" and (method_name == "GD_dichotomy" or method_name == "GD_golden"):
            # x0_test = np.array([-5, 5])
        # else:
            # x0_test = np.array([params['x0_0'], params['x0_1']])
        x0_test = np.array([params['x0_0'], params['x0_1']])
        if method_name == "GD_constant":
            x_min = method(func, x0_test.copy(),
                           step=params['step'],
                           max_iter=params['max_iter'],
                           eps=params['eps'])

        elif method_name == "GD_decreasing":
            x_min = method(func, x0_test.copy(),
                           step=params['init_step'],
                           max_iter=params['max_iter'],
                           eps=params['eps'])

        elif method_name == "GD_armijo":
            x_min = method(func, x0_test.copy(),
                           step=params['init_step'],
                           max_iter=params['max_iter'],
                           eps=params['eps'])

        elif method_name == "GD_wolfe":
            x_min = method(func, x0_test.copy(),
                           step=params['init_step'],
                           max_iter=params['max_iter'],
                           eps=params['eps'])

        elif method_name == "GD_golden":
            x_min = method(func, x0_test.copy(),
                            a=params['a'],
                            b=params['b'],
                        #    a=0,
                        #    b=0.1,
                           max_iter=params['max_iter'],
                           eps=params['eps'])

        elif method_name == "GD_dichotomy":
            x_min = method(func, x0_test.copy(),
                           a=params['a'],
                           b=params['b'],
                        #    a=0,
                        #    b=0.1,
                           max_iter=params['max_iter'],
                           eps=params['eps'])

        print_result(method_name, x_min)
        print(f"Function value: {func(x_min):.6f}")
        print(f"Start point: {x0_test}")


for func_name, func, sympy_func in test_functions_sympy:
    print(f"\nFunction: {func_name}")
    # if func_name == "Rosenbrock":
    #     x0_test = np.array([-1.5, 1.5])
    # else:
    #     x0_test = np.array([15.0, -15.0])
    for method_name, method, _ in methods_lab2:
        if method_name not in best_params[func_name]:
            continue
        params = best_params[func_name][method_name]
        x0_test = np.array([params['x0_0'], params['x0_1']])
        print(f"\nMethod: {method_name}")

        if method_name == "Newton_wolfe":
            x_min, _ = method(func, sympy_func, [x, y],
                             x0=x0_test.copy(),
                             max_iter=params['max_iter'],
                             eps=params['eps'])
        elif method_name == "Newton_golden":
            x_min, _ = method(func, sympy_func, [x, y],
                             x0=x0_test.copy(),
                             max_iter=params['max_iter'],
                             eps=params['eps'])
        elif method_name == "BFGS":
            x_min, _ = method(func,
                             x0=x0_test.copy(),
                             max_iter=params['max_iter'],
                             eps=params['eps'])

        print_result(method_name, x_min)
        print(f"Function value: {func(x_min):.6f}")
        print(f"Start point: {x0_test}")
