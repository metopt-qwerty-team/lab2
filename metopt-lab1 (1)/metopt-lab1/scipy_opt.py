from scipy.optimize import minimize, line_search, golden
from methods import *

x0_f1 = np.array([15.0, -15.0])
x0_f2 = np.array([15.0, -15.0])
x0_f3 = np.array([15.0, -15.0])

f1 = lambda x: x[0] ** 2 + x[1] ** 2
f2 = lambda x: 1000 * (x[0] ** 2) + x[1] ** 2


def grad_f1(x):
    return gradient(f1, x)


def grad_f2(x):
    return gradient(f2, x)


x0 = np.array([15.0, -15.0])

methods = ['BFGS']

for method in methods:
    print(f"\nMethod: {method}")

    res = minimize(f2, x0, method=method, jac=grad_f2)
    print("iterations: ", res.nit)
    print_result("BFGS", res.x)


def scipy_line_search_gd(f, grad_func, x0, max_iter=1000, eps=1e-6):
    x = x0.copy()
    for k in range(max_iter):
        g = grad_func(x)
        if np.linalg.norm(g) < eps:
            break
        direction = -g

        alpha = line_search(f, grad_func, x, direction)[0]

        x = x + alpha * direction
    print("Wolfe iterations: ", k)
    return x


def scipy_golden_gd(f, x0, max_iter=1000, eps=1e-6):
    x = x0.copy()
    for k in range(max_iter):
        grad = gradient(f, x)
        if np.linalg.norm(grad) < eps:
            break

        phi = lambda a: f(x - a * grad)

        alpha = golden(phi, brack=(0, 1))

        x = x - alpha * grad
    print("Golden iterations: ", k)
    return x


x0 = np.array([15.0, -15.0])

print_result("Wolfe", scipy_line_search_gd(f2, grad_f2, x0_f2))
print_result("Golden section", scipy_golden_gd(f2, x0_f2))
