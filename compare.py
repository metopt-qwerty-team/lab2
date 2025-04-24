import numpy as np

from methods import newton_method_golden_section, newton_method_wolfe_step, bfgs_method
from methods_from_lab1 import gradient_descent_with_constant_step
from utils import print_result

import sympy as sp

x, y = sp.symbols('x y')

start_points = {
    "rosenbrock": np.array([1.5, 1.5]),
    "himmelblau": np.array([-4.0, 4.0]),
    "quadratic": np.array([15.0, -15.0])
}

max_iter = 1000
eps = 1e-6


def compare_methods_for_function(func_name, func, func_sympy, start_point):
    print(f"=== Сравнение методов для функции {func_name} ===")
    print(f"Начальная точка: {start_point}")
    print("\nПользовательские методы:")

    print("1. Метод Ньютона с золотым сечением:")
    x_min, _ = newton_method_golden_section(
        func,
        func_sympy,
        variables=[x, y],
        x0=start_point,
        max_iter=max_iter,
        eps=eps
    )
    print_result("", x_min)
    print(f"Значение функции: {func(x_min):.6f}")

    print("\n2. Метод Ньютона с правилом Вульфа:")
    x_min, _ = newton_method_wolfe_step(
        func,
        func_sympy,
        variables=[x, y],
        x0=start_point,
        max_iter=max_iter,
        eps=eps
    )
    print_result("", x_min)
    print(f"Значение функции: {func(x_min):.6f}")

    print("\n3. Пользовательский метод BFGS:")
    x_min, _ = bfgs_method(
        func,
        x0=start_point,
        max_iter=max_iter,
        eps=eps
    )
    print_result("", x_min)
    print(f"Значение функции: {func(x_min):.6f}")

    print("\n4. Градиентный спуск с постоянным шагом:")
    if func_name == "квадратичная с высокой обусловленностью":
        step = 0.0001
    else:
        step = 0.001
    x_min = gradient_descent_with_constant_step(
        func,
        start_point.copy(),
        step=step,
        max_iter=max_iter,
        eps=eps
    )
    print_result("", x_min)
    print(f"Значение функции: {func(x_min):.6f}")
