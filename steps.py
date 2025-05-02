from utils import *


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

    print((a + b) / 2)
    return (a + b) / 2


def wolfe_step(f, x_k, grad, d_k, tracker, a=1, c_1=0.1, c_2=0.9, max_iter=20):
    f_x_k = f(x_k)
    tracker.f += 1
    grad_x_k = np.dot(grad, d_k)
    left, right = 0, np.inf

    for i in range(max_iter):
        x_new = x_k + a * d_k
        f_new = f(x_new)
        tracker.f += 1
        grad_new = gradient(f, x_new, tracker)
        grad_new_d = np.dot(grad_new, d_k)

        if f_new > f_x_k + c_1 * a * grad_x_k:
            right = a
            a = (left + right) / 2
        elif grad_new_d < c_2 * grad_x_k:
            left = a
            a = 2 * a if right == np.inf else (left + right) / 2
        else:
            # print(a)
            return a, i

    print(a)
    return a, max_iter


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
