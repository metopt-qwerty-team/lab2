import numpy as np
import sympy as sp
from scipy.optimize import approx_fprime, minimize


# def gradient(f, x, eps=1e-8):
#     return approx_fprime(x, f, eps)


def gradient(f, x, eps=1e-8):
    grad = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        x1, x2 = np.array(x, dtype=float), np.array(x, dtype=float)
        x1[i] += eps
        x2[i] -= eps
        grad[i] = (f(x1) - f(x2)) / (2 * eps)
    return grad


def symbolic_hessian(f_sympy, variables):
    hess = sp.hessian(f_sympy, variables)
    hess_func = sp.lambdify(variables, hess, 'numpy')
    return lambda x: np.array(hess_func(*x), dtype=float)


def dichotomy_step(f, x, direction, a=0, b=1, eps=1e-5, c=10):
    while abs(b - a) > eps:
        delta = (b - a) / c
        x1 = (a + b) / 2 - delta
        x2 = (a + b) / 2 + delta

        y1 = f(x + x1 * direction)
        y2 = f(x + x2 * direction)

        if y1 > y2:
            a = x1
        else:
            b = x2

    return (a + b) / 2


def newton_method_dichotomy(f, f_sympy, variables, x0, max_iter=100, eps=1e-6):
    hessian_func = symbolic_hessian(f_sympy, variables)

    x = x0.copy()
    for k in range(max_iter):
        grad = gradient(f, x)
        if np.linalg.norm(grad) < eps:
            break

        H = hessian_func(x)
        try:
            direction = np.linalg.solve(H, -grad)
        except np.linalg.LinAlgError:
            H += 1e-5 * np.eye(len(x))
            direction = np.linalg.solve(H, -grad)

        alpha = dichotomy_step(f, x, direction)
        x = x + alpha * direction

    print(f"Newton method with dichotomy: iterations: {k}")
    return x


x, y = sp.symbols('x y')
f_sympy = 100 * (y - x ** 2) ** 2 + (1 - x) ** 2


def rosenbrock(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


x0 = np.array([-1.5, 1.5])
# x0 = np.array([-15, 15])

x_min = newton_method_dichotomy(
    f=rosenbrock,
    f_sympy=f_sympy,
    variables=[x, y],
    x0=x0
)

print("minimum:", x_min)


def scipy_newton_cg(f, x0, eps=1e-6, max_iter=100):
    res = minimize(f, x0, method='Newton-CG',
                   jac=lambda x: gradient(f, x),
                   options={'xtol': eps})
    print(f"Newton-CG: {res.message}, iterations: {res.nit}")
    return res.x


def scipy_bfgs(f, x0, eps=1e-6, max_iter=100):
    res = minimize(f, x0, method='BFGS',
                   jac=lambda x: gradient(f, x),
                   options={'gtol': eps})
    print(f"BFGS: {res.message}, iterations: {res.nit}")
    return res.x


def scipy_lbfgs(f, x0, eps=1e-6, max_iter=100):
    res = minimize(f, x0, method='L-BFGS-B',
                   jac=lambda x: gradient(f, x),
                   options={'ftol': eps})
    print(f"L-BFGS: {res.message}, iterations: {res.nit}")
    return res.x


methods = {
    "Newton-CG": scipy_newton_cg,
    "BFGS": scipy_bfgs,
    "L-BFGS": scipy_lbfgs
}

print("==============SCIPY OPTIMIZE METHODS==============")
for name, method in methods.items():
    x_min = method(rosenbrock, x0)
    print(f"{name} found minimum at: {x_min}")
    print(f"Function value: {rosenbrock(x_min):.6f}")
    print("-" * 50)
