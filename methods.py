from scipy.optimize import minimize
from steps import *
from utils import *


def newton_method_wolfe_step(f, f_sympy, variables, x0, max_iter=1000, eps=1e-6, wolfe_params={'a': 1, 'c1': 0.1, 'c2': 0.9}):
    tracker = Tracker()
    hessian_func = symbolic_hessian(f_sympy, variables)
    x = x0.copy()
    trajectory = [x.copy()]

    x = x0.copy()

    iteration = 0

    for k in range(max_iter):
        grad = gradient(f, x, tracker)
        if np.linalg.norm(grad) < eps:
            break

        H = hessian_func(x, tracker)

        min_eigval = np.linalg.eigvals(H).min()
        if min_eigval <= 0:
            H += (abs(min_eigval) + 1e-5) * np.eye(len(x))

        direction = np.linalg.solve(H, -grad)

        alpha, i = wolfe_step(f, x, grad, direction, tracker, wolfe_params['a'], wolfe_params['c1'], wolfe_params['c2'])
        iteration += i
        x = x + alpha * direction
        iteration += 1
        trajectory.append(x.copy())

    # print(
        # f"Newton method with wolfe rule: iterations: {iteration}, grad: {tracker.g}, hess: {tracker.h}, f: {tracker.f}")
    return x, np.array(trajectory)


def newton_method_golden_section(f, f_sympy, variables, x0, max_iter=1000, eps=1e-6, golden_params = {'a': 0, 'b': 1}):
    tracker = Tracker()
    hessian_func = symbolic_hessian(f_sympy, variables)

    x = x0.copy()
    trajectory = [x.copy()]

    for k in range(max_iter):
        grad = gradient(f, x, tracker)
        if np.linalg.norm(grad) < eps:
            break

        H = hessian_func(x, tracker)

        min_eigval = np.linalg.eigvals(H).min()
        if min_eigval <= 0:
            H += (abs(min_eigval) + 1e-5) * np.eye(len(x))

        direction = np.linalg.solve(H, -grad)
        alpha = golden_section(f, golden_params['a'], golden_params['b'], x, direction, tracker)
        x = x + alpha * direction
        trajectory.append(x.copy())

    # print(f"Newton method with golden section: iterations: {k}, grad: {tracker.g}, hes: {tracker.h}, f: {tracker.f}")
    return x, np.array(trajectory)


def bfgs_method(f, x0, max_iter=100, eps=1e-6, golden_params = {'a': 0, 'b': 1}):
    tracker = Tracker()
    n = len(x0)
    H = np.eye(n)
    x = x0.copy()
    trajectory = [x.copy()]

    for k in range(max_iter):
        grad = gradient(f, x, tracker)

        if np.linalg.norm(grad) < eps:
            break

        direction = -H @ grad

        alpha = golden_section(f, golden_params['a'], golden_params['b'], x, direction, tracker)

        x_new = x + alpha * direction

        grad_new = gradient(f, x_new, tracker)
        y = grad_new - grad
        s = x_new - x

        if y.T @ s <= 0:
            x = x_new
            trajectory.append(x.copy())
            continue

        rho = 1.0 / (y.T @ s)
        I = np.eye(n)
        term1 = I - rho * np.outer(s, y)
        term2 = I - rho * np.outer(y, s)
        H = term1 @ H @ term2 + rho * np.outer(s, s)

        tracker.h += 1
        x = x_new
        trajectory.append(x.copy())

    # print(f"BFGS method: iterations: {k}, grad: {tracker.g}, hess: {tracker.h}, f: {tracker.f}")
    return x, np.array(trajectory)


def scipy_newton_cg(f, x0, eps=1e-6, max_iter=100, callback=None):
    res = minimize(f, x0, method='Newton-CG',
                   jac=lambda x: gradient(f, x, Tracker()),
                   options={'xtol': eps, 'maxiter': max_iter},
                   callback=callback)
    print(f"Newton-CG: {res.message}, iterations: {res.nit}, func calc: {res.nfev}, grad: {res.njev}, hes: {res.nhev}")
    return res.x


def scipy_bfgs(f, x0, eps=1e-6, max_iter=100, callback=None):
    res = minimize(f, x0, method='BFGS',
                   jac=lambda x: gradient(f, x, Tracker()),
                   options={'gtol': eps, 'maxiter': max_iter},
                   callback=callback)
    print(f"BFGS: {res.message}, iterations: {res.nit}, func calc: {res.nfev}, grad: {res.njev}")
    return res.x


def scipy_lbfgs(f, x0, eps=1e-6, max_iter=100, callback=None):
    res = minimize(f, x0, method='L-BFGS-B',
                   jac=lambda x: gradient(f, x, Tracker()),
                   options={'ftol': eps, 'maxiter': max_iter},
                   callback=callback)
    print(f"L-BFGS: {res.message}, iterations: {res.nit}, func calc: {res.nfev}, grad: {res.njev}")
    return res.x
