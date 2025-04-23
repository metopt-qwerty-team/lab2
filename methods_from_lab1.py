import numpy as np
from methods import gradient
from main import Tracker


def gradient_descent_with_constant_step(f, x, step=0.001, max_iter=10000, eps=1e-6):
    tracker = Tracker()
    for k in range(max_iter):
        grad = gradient(f, x, tracker)
        if np.linalg.norm(grad) < eps:
            break
        x = x - step * grad
    print(f"gradient_descent_with_constant_step: iterations: {k}, grad: {tracker.g}, hes: {tracker.h}, f: {tracker.f}")
    return x


def gradient_descent_with_decreasing_step(f, x, step=1, max_iter=10000, eps=1e-6):
    tracker = Tracker()
    for k in range(1, max_iter + 1):
        grad = gradient(f, x, tracker)
        if np.linalg.norm(grad) < eps:
            break
        # step = step / k
        x = x - step / k * grad
    print(f"gradient_descent_with_decreasing_step: iterations: {k}, grad: {tracker.g}, hes: {tracker.h}, f: {tracker.f}")
    return x


def armijo_step(f, grad, x_k, tracker, a=1.0, b=0.5, c=0.1):
    d_k = -grad
    tracker.f += 2
    while f(x_k + a * d_k) > f(x_k) + c * a * np.dot(grad, d_k) and a > 1e-8:
        a *= b
        tracker.f += 2
    return a


def wolfe_step(f, grad, x_k, tracker, a=1, c_1=0.1, c_2=0.9, max_iter=20):
    f_x_k = f(x_k)
    tracker.f += 1
    d_k = -grad
    grad_x_k = np.dot(grad, d_k)
    left, right = 0, np.inf
    for i in range(max_iter):
        current_x = x_k + a * d_k
        current_f = f(current_x)
        tracker.f += 1
        current_grad = gradient(f, current_x, tracker)
        current_grad_k = np.dot(current_grad, d_k)

        if current_f > f_x_k + c_1 * a * grad_x_k:
            right = a
            a = (left + right) / 2
        elif abs(current_grad_k) > c_2 * abs(grad_x_k):
            left = a
            if right == np.inf:
                a *= 2
            else:
                a = (left + right) / 2
        else:
            return a, i

    return a, i


def gradient_descent_armijo(f, x, step=1.0, b=0.5, c=0.1, max_iter=10000, eps=1e-6):
    tracker = Tracker()
    for k in range(max_iter):
        grad = gradient(f, x, tracker)
        if np.linalg.norm(grad) < eps:
            break
        step = armijo_step(f, grad, x, tracker, step, b, c)
        x = x - step * grad
    print(f"gradient_descent_armijo: iterations: {k}, grad: {tracker.g}, hes: {tracker.h}, f: {tracker.f}")
    return x


def gradient_descent_wolfe(f, x, step=1.0, c1=0.1, c2=0.9, max_iter=10000, eps=1e-6):
    tracker = Tracker()
    for k in range(max_iter):
        grad = gradient(f, x, tracker)
        if np.linalg.norm(grad) < eps:
            break
        step, cnt = wolfe_step(f, grad, x, tracker, step, c1, c2)  # cnt ??????
        x = x - step * grad
    print(f"gradient_descent_wolfe: iterations: {k}, grad: {tracker.g}, hes: {tracker.h}, f: {tracker.f}")
    return x


def golden_section(f, a, b, x, direction, tracker, eps=1e-5):
    direction = -direction
    c = 2 / (1 + np.sqrt(5))

    x1 = b - (b - a) * c
    x2 = a + (b - a) * c

    y1 = f(x - x1 * direction)
    y2 = f(x - x2 * direction)
    tracker.f += 2

    while abs(b - a) > eps:
        if y1 < y2:
            b, x2, y2 = x2, x1, y1
            x1 = b - (b - a) * c
            y1 = f(x - x1 * direction)
        else:
            a, x1, y1 = x1, x2, y2
            x2 = a + (b - a) * c
            y2 = f(x - x2 * direction)
        tracker.f += 1

    return (a + b) / 2


def gradient_descent_with_golden_section(f, x, a=0, b=0.1, max_iter=10000, eps=1e-6):
    tracker = Tracker()
    for k in range(max_iter):
        grad = gradient(f, x, tracker)
        if np.linalg.norm(grad) < eps:
            break
        step = golden_section(f, a, b, x, grad, tracker)
        x = x - step * grad
    print(f"gradient_descent_with_golden_section: iterations: {k}, grad: {tracker.g}, hes: {tracker.h}, f: {tracker.f}")
    return x


# c > 2
def dichotomy_step(f, x, grad, tracker, a=0.0, b=1.0, eps=1e-5, c=10):
    while abs(b - a) > eps:
        # delta in (0, (b - a) / 2)
        delta = (b - a) / c
        x1 = (a + b) / 2 - delta
        x2 = (a + b) / 2 + delta

        y1 = f(x - x1 * grad)
        y2 = f(x - x2 * grad)
        tracker.f += 2

        if y1 > y2:
            a = x1
        else:
            b = x2

    return (a + b) / 2


def gradient_descent_dichotomy(f, x, a=0, b=0.1, max_iter=10000, eps=1e-6):
    tracker = Tracker()
    for k in range(max_iter):
        grad = gradient(f, x, tracker)
        if np.linalg.norm(grad) < eps:
            break
        step = dichotomy_step(f, x, grad, tracker, a, b)
        x = x - step * grad
    print(f"gradient_descent_dichotomy: iterations: {k}, grad: {tracker.g}, hes: {tracker.h}, f: {tracker.f}")
    return x


def print_result(method_name, result):
    print(f"{method_name}: \n \t \t \t [{result[0]:.16f}, {result[1]:.16f}]")
