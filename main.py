from methods import *
from methods_from_lab1 import *

x, y = sp.symbols('x y')
rosenbrock_sympy = 100 * (y - x ** 2) ** 2 + (1 - x) ** 2
himmelblau_sympy = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
f_sympy = 1000 * x ** 2 + y ** 2


def rosenbrock(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


def himmelblau(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


def f(x):
    return 1000 * (x[0] ** 2) + x[1] ** 2


start_points = np.array([
    # [-4.0, -4.0]
    # [-4.0, 4.0],
    # [2.0, -4.0],
    # [15.0, -15.0]
    [1.5, 1.5]
]
)

test_func = [
    ("Rosenbrock", rosenbrock, rosenbrock_sympy, start_points, 1000),
    # ("Himmelblau", himmelblau, himmelblau_sympy, start_points, 1000),
    # ("1000x^2 + y^2", f, f_sympy, start_points, 1000),
]

for func_name, func, func_sympy, start_point, max_iter in test_func:
    for x0 in start_point:
        print("newton_method_golden_section", x0)
        x_min = newton_method_golden_section(
            func,
            func_sympy,
            variables=[x, y],
            x0=x0,
            max_iter=max_iter
        )
        print_result(func_name, x_min)
        print(f"Function value: {func(x_min):.6f}")
        print("-" * 50)

        print("newton_method_wolfe_step", x0)
        x_min = newton_method_wolfe_step(
            func,
            func_sympy,
            variables=[x, y],
            x0=x0,
            max_iter=max_iter
        )
        print_result(func_name, x_min)
        print(f"Function value: {func(x_min):.6f}")
        print("-" * 50)
    print("=" * 100)

print("==============CUSTOM BFGS METHOD==============")
for func_name, func, func_sympy, start_points, max_iter in test_func:
    for x0 in start_points:
        print("-" * 50)
        x_min = bfgs_method(func, x0, max_iter=max_iter)
        print_result(f"Custom BFGS ({func_name})", x0)
        print("Min:", x_min)
        print(f"Function value: {func(x_min):.6f}")

x0 = np.array([-15, 15])

methods = [
    ("Newton-CG", scipy_newton_cg, [rosenbrock, himmelblau, f], x0),
    ("BFGS", scipy_bfgs, [rosenbrock, himmelblau, f], x0),
    ("L-BFGS", scipy_lbfgs, [rosenbrock, himmelblau, f], x0)
]

print("==============SCIPY OPTIMIZE METHODS==============")
for name, method, functions, start_point in methods:
    print("-" * 50)
    for func in functions:
        x_min = method(func, start_point)
        print_result(name, x_min)
        # print(f"{name} found minimum at: {x_min} for function: {func.__name__}")
        print(f"Function value: {func(x_min):.6f} for function: {func.__name__}")
        print("-" * 25)
    print("-" * 50)

print("=================GD METHODS======================")

test_functions = [
    ("Rosenbrock", rosenbrock),
    # ("Himmelblau", himmelblau),
    # ("1000x^2 + y^2", f)
]

methods = [
    ("GD_constant", gradient_descent_with_constant_step),
    ("GD_decreasing", gradient_descent_with_decreasing_step),
    ("GD_armijo", gradient_descent_armijo),
    ("GD_wolfe", gradient_descent_wolfe),
    ("GD_golden", gradient_descent_with_golden_section),
    ("GD_dichotomy", gradient_descent_dichotomy)
]

for func_name, func in test_functions:
    for x0 in start_points:
        for method_name, method in methods:
            print(func_name, "start point:", x0)
            result = method(func, x0.copy())
            print(f"{method_name:20s} → minimum: [{result[0]:.8f}, {result[1]:.8f}] ")
            print(f"Function value: {func(result):.6f}")
            print("-" * 50)
