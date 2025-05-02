import numpy as np

# Импорт методов из существующих файлов без дублирования их реализации
from main import rosenbrock, himmelblau, f as quadratic
from main import rosenbrock_sympy, himmelblau_sympy, f_sympy as quadratic_sympy
from methods import newton_method_golden_section, newton_method_wolfe_step, bfgs_method
from methods import scipy_newton_cg, scipy_bfgs, scipy_lbfgs
from methods_from_lab1 import gradient_descent_with_constant_step, gradient_descent_armijo, gradient_descent_wolfe
from utils import print_result

# Используем символические переменные из main.py через импорт
import sympy as sp
x, y = sp.symbols('x y')

# Начальные точки для каждой функции
start_points = {
    "rosenbrock": np.array([1.5, 1.5]),
    "himmelblau": np.array([-4.0, 4.0]),
    "quadratic": np.array([15.0, -15.0])
}

max_iter = 1000
eps = 1e-6

# Функция для сравнения пользовательских и библиотечных методов
def compare_methods_for_function(func_name, func, func_sympy, start_point):
    print(f"=== Сравнение методов для функции {func_name} ===")
    print(f"Начальная точка: {start_point}")
    print("\nПользовательские методы:")

    # Пользовательский метод Ньютона с золотым сечением
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

    # Пользовательский метод Ньютона с правилом Вульфа
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

    # Пользовательский метод BFGS
    print("\n3. Пользовательский метод BFGS:")
    x_min, _ = bfgs_method(
        func,
        x0=start_point,
        max_iter=max_iter,
        eps=eps
    )
    print_result("", x_min)
    print(f"Значение функции: {func(x_min):.6f}")

    # Градиентный спуск с постоянным шагом
    print("\n4. Градиентный спуск с постоянным шагом:")
    if func_name == "квадратичная с высокой обусловленностью":
        step = 0.0001  # Маленький шаг для функции с высокой обусловленностью
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