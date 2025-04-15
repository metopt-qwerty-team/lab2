import numpy as np
import sympy as sp
from scipy.optimize import approx_fprime, minimize
from steps import *
from methods import *

# def gradient(f, x, eps=1e-8):
#     return approx_fprime(x, f, eps)

class Tracker:
    def __init__(self, g=0, h=0):
        self._g = g
        self._h = h

    @property
    def g(self):
        return self._g
    @property
    def h(self):
        return self._h
    @g.setter
    def g(self, value):
        self._g = value
    @h.setter
    def h(self, value):
        self._h = value



def newton_method_golden_section(f, f_sympy, variables, x0, max_iter=100, eps=1e-6):
    tracker = Tracker()
    hessian_func = symbolic_hessian(f_sympy, variables)

    x = x0.copy()
    
    for k in range(max_iter):
        grad = gradient(f, x, tracker)
        if np.linalg.norm(grad) < eps:
            break

        H = hessian_func(x, tracker)
        try:
            direction = np.linalg.solve(H, -grad)
        except np.linalg.LinAlgError:
            H += 1e-5 * np.eye(len(x))
            direction = np.linalg.solve(H, -grad)

        # alpha, _ = wolfe_step(f, x, grad, direction, tracker)
        # alpha = dichotomy_step(f, x, direction)
        alpha = golden_section(f, 0, 1, x, direction)
        # print("alpha: ", alpha)
        x = x + alpha * direction

    print(f"Newton method with golden section: iterations: {k}, grad: {tracker.g}, hes: {tracker.h}")
    return x


def bfgs_method(f, x0, max_iter=100, eps=1e-6):
    n = len(x0)
    H = np.eye(n)  # Начальное приближение обратной матрицы Гессиана
    x = x0.copy()
    tracker = Tracker()
    
    for k in range(max_iter):
        grad = gradient(f, x, tracker)
        
        # Критерий остановки
        if np.linalg.norm(grad) < eps:
            break
            
        # Направление спуска
        direction = -H @ grad
        
        # Выбор шага с помощью золотого сечения
        alpha = golden_section(f, 0, 1, x, direction)
        
        # Обновление точки
        x_new = x + alpha * direction
        
        # Вычисление изменений
        grad_new = gradient(f, x_new, tracker)
        y = grad_new - grad
        s = x_new - x
        
        # Проверка условия кривизны (должно выполняться y.T @ s > 0)
        if y.T @ s <= 0:
            # Если условие не выполняется, пропускаем обновление H
            x = x_new
            continue
            
        # Обновление матрицы H по формуле BFGS
        rho = 1.0 / (y.T @ s)
        I = np.eye(n)
        term1 = I - rho * np.outer(s, y)
        term2 = I - rho * np.outer(y, s)
        H = term1 @ H @ term2 + rho * np.outer(s, s)
        
        tracker.h += 1  # Учитываем обновление приближения Гессиана
        x = x_new
    
    print(f"BFGS method: iterations: {k}, grad: {tracker.g}, hess_updates: {tracker.h}")
    return x



x, y = sp.symbols('x y')
rosenbrock_sympy = 100 * (y - x ** 2) ** 2 + (1 - x) ** 2
himmelblau_sympy = (x ** 2 + y - 11) ** 2 + (x + y**2 - 7) ** 2
f_sympy = 1000 * x ** 2 + y ** 2


def rosenbrock(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

symbolic_hessian

def himmelblau(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

def f(x):
    return 1000 * (x[0] ** 2) + x[1]**2

x0 = np.array([-1.5, 1.5])

test_func = [
    ("Rosenbrock", rosenbrock, rosenbrock_sympy, x0, 1000),
    ("Himmelblau", himmelblau, himmelblau_sympy, x0, 1000),
    ("Func", f, f_sympy, x0, 1000),

]

for func_name, func, func_sympy, start_point, max_iter in test_func:
    tracker = Tracker()
    print("newton_method_golden_section")
    x_min = newton_method_golden_section(
        func, 
        func_sympy,
        variables=[x, y],
        x0=start_point,
        max_iter=max_iter
    )
    print_result(func_name, x_min)

print("==============CUSTOM BFGS METHOD==============")
for func_name, func, func_sympy, start_point, max_iter in test_func:
    print("-" * 50)
    x_min = bfgs_method(func, start_point, max_iter=max_iter)
    print_result(f"Custom BFGS ({func_name})", x_min)
    print(f"Function value: {func(x_min):.6f}")
    print("-" * 50)


# x0 = np.array([-15, 15])


# x_min = newton_method_dichotomy(
#     f=func,
#     f_sympy=f_sympy,
#     variables=[x, y],
#     x0=x0,
#     max_iter=100000
# )
# print_result(x_min)
# print("minimum:", x_min)


tracker = Tracker()
def scipy_newton_cg(f, x0, eps=1e-6, max_iter=100):
    res = minimize(f, x0, method='Newton-CG',
                   jac=lambda x: gradient(f, x, tracker),
                   options={'xtol': eps})
    print(f"Newton-CG: {res.message}, iterations: {res.nit}, func calc: {res.nfev}, grad: {res.njev}, hes: {res.nhev}")
    return res.x


def scipy_bfgs(f, x0, eps=1e-6, max_iter=100):
    res = minimize(f, x0, method='BFGS',
                   jac=lambda x: gradient(f, x, tracker),
                   options={'gtol': eps})
    print(f"BFGS: {res.message}, iterations: {res.nit}, func calc: {res.nfev}, grad: {res.njev}")
    return res.x


def scipy_lbfgs(f, x0, eps=1e-6, max_iter=100):
    res = minimize(f, x0, method='L-BFGS-B',
                   jac=lambda x: gradient(f, x, tracker),
                   options={'ftol': eps})
    print(f"L-BFGS: {res.message}, iterations: {res.nit}, func calc: {res.nfev}, grad: {res.njev}")
    return res.x


methods = [
    ("Newton-CG",  scipy_newton_cg, [rosenbrock, himmelblau, f], x0),
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