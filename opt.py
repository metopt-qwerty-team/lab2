import numpy as np
import optuna
from methods_from_lab1 import *
from main import *

optuna.logging.set_verbosity(optuna.logging.WARNING)  

def rosenbrock(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

def himmelblau(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

def quadratic(x):
    return 1000 * (x[0] ** 2) + x[1]**2


x0 = np.array([-1.5, 1.5])

def objective_gd_constant(trial):
    step = trial.suggest_float('step', 1e-6, 1.0, log=True)
    eps = trial.suggest_float('eps', 1e-8, 1e-4, log=True)
    max_iter = trial.suggest_int('max_iter', 100, 10000)
    
    x_min = gradient_descent_with_constant_step(rosenbrock, x0.copy(), 
                                              step=step, max_iter=max_iter, eps=eps)
    return rosenbrock(x_min)

def objective_gd_decreasing(trial):
    init_step = trial.suggest_float('init_step', 0.1, 10.0)
    eps = trial.suggest_float('eps', 1e-8, 1e-4, log=True)
    max_iter = trial.suggest_int('max_iter', 100, 10000)
    
    x_min = gradient_descent_with_decreasing_step(rosenbrock, x0.copy(),
                                                step=init_step, max_iter=max_iter, eps=eps)
    return rosenbrock(x_min)

def objective_gd_armijo(trial):
    init_step = trial.suggest_float('init_step', 0.1, 10.0)
    b = trial.suggest_float('b', 0.1, 0.9)
    c = trial.suggest_float('c', 0.01, 0.49)
    eps = trial.suggest_float('eps', 1e-8, 1e-4, log=True)
    max_iter = trial.suggest_int('max_iter', 100, 10000)
    
    x_min = gradient_descent_armijo(rosenbrock, x0.copy(),
                                  step=init_step, max_iter=max_iter, eps=eps)
    return rosenbrock(x_min)

def objective_gd_wolfe(trial):
    init_step = trial.suggest_float('init_step', 0.1, 10.0)
    c1 = trial.suggest_float('c1', 0.01, 0.49)
    c2 = trial.suggest_float('c2', 0.5, 0.99)
    eps = trial.suggest_float('eps', 1e-8, 1e-4, log=True)
    max_iter = trial.suggest_int('max_iter', 100, 10000)
    
    x_min = gradient_descent_wolfe(rosenbrock, x0.copy(),
                                 step=init_step, max_iter=max_iter, eps=eps)
    return rosenbrock(x_min)

def objective_gd_golden(trial):
    a = trial.suggest_float('a', 0.0, 0.1)
    b = trial.suggest_float('b', 0.1, 1.0)
    eps = trial.suggest_float('eps', 1e-8, 1e-4, log=True)
    max_iter = trial.suggest_int('max_iter', 100, 10000)
    
    x_min = gradient_descent_with_golden_section(rosenbrock, x0.copy(),
                                               a=a, b=b, max_iter=max_iter, eps=eps)
    return rosenbrock(x_min)

def objective_gd_dichotomy(trial):
    a = trial.suggest_float('a', 0.0, 0.1)
    b = trial.suggest_float('b', 0.1, 1.0)
    c = trial.suggest_float('c', 2.1, 20.0)
    eps = trial.suggest_float('eps', 1e-8, 1e-4, log=True)
    max_iter = trial.suggest_int('max_iter', 100, 10000)
    
    x_min = gradient_descent_dichotomy(rosenbrock, x0.copy(),
                                     a=a, b=b, max_iter=max_iter, eps=eps)
    return rosenbrock(x_min)

methods = [
    ("GD_constant", objective_gd_constant),
    ("GD_decreasing", objective_gd_decreasing),
    ("GD_armijo", objective_gd_armijo),
    ("GD_wolfe", objective_gd_wolfe),
    ("GD_golden", objective_gd_golden),
    ("GD_dichotomy", objective_gd_dichotomy)
]

best_params = {}
for method_name, objective in methods:
    print(f"\nOptimizing {method_name}...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    best_params[method_name] = study.best_params
    print(f"Best params for {method_name}:")
    print(study.best_params)
    print(f"Best value: {study.best_value}")

print("\nTesting with optimized parameters...")
test_functions = [("Rosenbrock", rosenbrock), ("Himmelblau", himmelblau), ("Quadratic", quadratic)]

for func_name, func in test_functions:
    print(f"\n=== Testing on {func_name} function ===")
    for method_name, params in best_params.items():
        print(f"\nMethod: {method_name}")
        x0_test = x0.copy()
        
        if method_name == "GD_constant":
            x_min = gradient_descent_with_constant_step(
                func, x0_test, 
                step=params['step'],
                max_iter=params['max_iter'],
                eps=params['eps']
            )
        elif method_name == "GD_decreasing":
            x_min = gradient_descent_with_decreasing_step(
                func, x0_test,
                step=params['init_step'],
                max_iter=params['max_iter'],
                eps=params['eps']
            )
        elif method_name == "GD_armijo":
            x_min = gradient_descent_armijo(
                func, x0_test,
                step=params['init_step'],
                max_iter=params['max_iter'],
                eps=params['eps']
            )
        elif method_name == "GD_wolfe":
            x_min = gradient_descent_wolfe(
                func, x0_test,
                step=params['init_step'],
                max_iter=params['max_iter'],
                eps=params['eps']
            )
        elif method_name == "GD_golden":
            x_min = gradient_descent_with_golden_section(
                func, x0_test,
                a=params['a'],
                b=params['b'],
                max_iter=params['max_iter'],
                eps=params['eps']
            )
        elif method_name == "GD_dichotomy":
            x_min = gradient_descent_dichotomy(
                func, x0_test,
                a=params['a'],
                b=params['b'],
                max_iter=params['max_iter'],
                eps=params['eps']
            )
        
        print_result(method_name, x_min)
        print(f"Function value: {func(x_min):.6f}")


















methods_advanced = [
    ("Newton", newton_method_golden_section, rosenbrock_sympy, [x, y]),
    ("BFGS", bfgs_method, None, None)
]

test_functions = [
    ("Rosenbrock", rosenbrock, rosenbrock_sympy),
    ("Himmelblau", himmelblau, himmelblau_sympy),
    ("Quadratic", quadratic, f_sympy)
]

best_params_advanced = {}

for method_name, method, sympy_func, variables in methods_advanced:
    print(f"\n=== Optimizing {method_name} method ===")
    
    def objective(trial):
        eps = trial.suggest_float('eps', 1e-8, 1e-4, log=True)
        max_iter = trial.suggest_int('max_iter', 50, 500)
        
        if method_name == "Newton":
            x_min = method(rosenbrock, sympy_func, variables, 
                          x0=x0, max_iter=max_iter, eps=eps)
        else: 
            x_min = method(rosenbrock, x0=x0, max_iter=max_iter, eps=eps)
        
        return rosenbrock(x_min)
    
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50, show_progress_bar=False)
    best_params_advanced[method_name] = study.best_params
    
    print(f"Best parameters for {method_name}:")
    print(study.best_params)
    print(f"Best value on Rosenbrock: {study.best_value}")

print("\n=== Testing optimized methods on all functions ===")
for func_name, func, sympy_func in test_functions:
    print(f"\nFunction: {func_name}")
    
    for method_name, method, _, variables in methods_advanced:
        params = best_params_advanced[method_name]
        print(f"\nMethod: {method_name} with optimized parameters")
        
        if method_name == "Newton":
            x_min = method(func, sympy_func, variables,
                         x0=x0,
                         max_iter=params['max_iter'],
                         eps=params['eps'])
        else:
            x_min = method(func,
                          x0=x0,
                          max_iter=params['max_iter'],
                          eps=params['eps'])
        
        print_result(method_name, x_min)
        print(f"Function value: {func(x_min):.6f}")

