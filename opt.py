import optuna
from methods_from_lab1 import *
from main import *

optuna.logging.set_verbosity(optuna.logging.WARNING)


def rosenbrock(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


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

methods = [
    ("GD_constant", gradient_descent_with_constant_step),
    ("GD_decreasing", gradient_descent_with_decreasing_step),
    ("GD_armijo", gradient_descent_armijo),
    ("GD_wolfe", gradient_descent_wolfe),
    ("GD_golden", gradient_descent_with_golden_section),
    ("GD_dichotomy", gradient_descent_dichotomy)
]

best_params = {func_name: {} for func_name, _ in test_functions}

for func_name, func in test_functions:
    print(f"\n=== Optimizing for {func_name} function ===")

    for method_name, method in methods:
        print(f"\nMethod: {method_name}")


        def objective(trial):
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
                x_min = method(func, x0.copy(), step=init_step, max_iter=max_iter, eps=eps)

            elif method_name == "GD_golden":
                a = trial.suggest_float('a', 0.0, 0.1)
                b = trial.suggest_float('b', 0.1, 1.0)
                eps = trial.suggest_float('eps', 1e-8, 1e-4, log=True)
                max_iter = trial.suggest_int('max_iter', 100, 10000)
                x_min = method(func, x0.copy(), a=a, b=b, max_iter=max_iter, eps=eps)

            elif method_name == "GD_dichotomy":
                a = trial.suggest_float('a', 0.0, 0.1)
                b = trial.suggest_float('b', 0.1, 1.0)
                eps = trial.suggest_float('eps', 1e-8, 1e-4, log=True)
                max_iter = trial.suggest_int('max_iter', 100, 10000)
                x_min = method(func, x0.copy(), a=a, b=b, max_iter=max_iter, eps=eps)

            return func(x_min)


        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=100)

        best_params[func_name][method_name] = study.best_params
        print(f"Best params: {study.best_params}")
        print(f"Best value: {study.best_value}")

print("\n=== Testing with optimized parameters ===")
x0_test = np.array([-1.5, 1.5])
for func_name, func in test_functions:
    print(f"\nFunction: {func_name}")

    for method_name, method in methods:
        params = best_params[func_name][method_name]
        print(f"\nMethod: {method_name}")

        # x0_test = np.array([params['x0_0'], params['x0_1']])

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
                           max_iter=params['max_iter'],
                           eps=params['eps'])

        elif method_name == "GD_dichotomy":
            x_min = method(func, x0_test.copy(),
                           a=params['a'],
                           b=params['b'],
                           max_iter=params['max_iter'],
                           eps=params['eps'])

        print_result(method_name, x_min)
        print(f"Function value: {func(x_min):.6f}")
        print(f"Start point: {x0_test}")

'''
x, y = sp.symbols('x y')
rosenbrock_sympy = 100 * (y - x ** 2) ** 2 + (1 - x) ** 2
himmelblau_sympy = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
quadratic_sympy = 1000 * x ** 2 + y ** 2

test_functions_sympy = [
    ("Rosenbrock", rosenbrock, rosenbrock_sympy),
    ("Himmelblau", himmelblau, himmelblau_sympy),
    ("Quadratic", quadratic, quadratic_sympy)
]

methods_advanced = [
    ("Newton", newton_method_golden_section),
    ("BFGS", bfgs_method)
]

for func_name, _, _ in test_functions_sympy:
    best_params[func_name].update({method_name: {} for method_name, _ in methods_advanced})

for func_name, func, sympy_func in test_functions_sympy:
    print(f"\n=== Optimizing advanced methods for {func_name} function ===")

    for method_name, method in methods_advanced:
        print(f"\nMethod: {method_name}")


        def objective(trial):
            eps = trial.suggest_float('eps', 1e-8, 1e-4, log=True)
            max_iter = trial.suggest_int('max_iter', 50, 500)

            if method_name == "Newton":
                x_min = method(func, sympy_func, [x, y],
                               x0=x0.copy(),
                               max_iter=max_iter,
                               eps=eps)
            else:
                x_min = method(func,
                               x0=x0.copy(),
                               max_iter=max_iter,
                               eps=eps)

            return func(x_min)


        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)

        best_params[func_name][method_name] = study.best_params
        print(f"Best params: {study.best_params}")
        print(f"Best value: {study.best_value}")

print("\n=== Testing advanced methods with optimized parameters ===")
for func_name, func, sympy_func in test_functions_sympy:
    print(f"\nFunction: {func_name}")

    for method_name, method in methods_advanced:
        params = best_params[func_name][method_name]
        print(f"\nMethod: {method_name}")

        if method_name == "Newton":
            x_min = method(func, sympy_func, [x, y],
                           x0=x0.copy(),
                           max_iter=params['max_iter'],
                           eps=params['eps'])
        else:
            x_min = method(func,
                           x0=x0.copy(),
                           max_iter=params['max_iter'],
                           eps=params['eps'])

        print_result(method_name, x_min)
        print(f"Function value: {func(x_min):.6f}")
'''
