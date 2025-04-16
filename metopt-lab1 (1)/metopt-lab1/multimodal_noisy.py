from scipy.optimize import golden
from methods import *


def scipy_golden_gd(f, x0, max_iter=1000, eps=1e-6):
    x = x0.copy()
    for _ in range(max_iter):
        grad = gradient(f, x)
        if np.linalg.norm(grad) < eps:
            break

        phi = lambda a: f(x - a * grad)

        alpha = golden(phi, brack=(0, 1))

        x = x - alpha * grad
    return x


f3 = lambda x: (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


def f4(x, noise_scale=0.01):
    noise = np.random.normal(scale=noise_scale)
    return f3(x) + noise


f5 = lambda x: lambda y: f4(y, x)

start_points = [[15.0, -15.0]]
scales = [0.0001, 0.001, 0.01]

for point in start_points:
    print(f"\nTesting point {point}:")
    for scale in scales:
        print(f"\n\tTesting scale {scale}:")
        for i in range(10):
            result = gradient_descent_with_golden_section(f5(scale), point, 0, 0.1, 10000)
            print_result(f"Run {i + 1}", result)
