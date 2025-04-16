import numpy as np

'''
smth like "central difference"
f'(x_i) = (f(x_i + eps) - f(x_i - eps)) / 2 * eps
'''


def gradient(f, x, eps=1e-5):
    grad = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        x1, x2 = np.array(x, dtype=float), np.array(x, dtype=float)
        x1[i] += eps
        x2[i] -= eps
        grad[i] = (f(x1) - f(x2)) / (2 * eps)
    return grad


def gradient_descent_with_constant_step(f, x, step=0.001, max_iter=10000, eps=1e-6):
    for k in range(max_iter):
        grad = gradient(f, x)
        if np.linalg.norm(grad) < eps:
            break
        x = x - step * grad
    print("gradient_descent_with_constant_step iterations: ", k)
    return x


def gradient_descent_with_decreasing_step(f, x, step=1, max_iter=10000, eps=1e-6):
    for k in range(1, max_iter + 1):
        grad = gradient(f, x)
        if np.linalg.norm(grad) < eps:
            break
        step = step / k
        x = x - step * grad
    print("gradient_descent_with_decreasing_step iterations: ", k)
    return x


# c = (0, 0.5)
'''
About armijo condition:

1) let's see function:
* d_k - gradient descent direction (vector from R^n)
* f - function
* a - argument

function: fi_k(a) = f(x_k + a*d_k)
2) differentiate the function
fi_k'(a) = grad_f(x_k + a*d_k)^T * d_k

3) fi'(0) = grad_f(x_k)^T * d_k < 0 because d_k - gradient descent direction

4) armijo condition:
fi_k(a) <= fi_k(0) + a * c_1 * fi_k'(0), where c_1 from (0, 0.5)
<=>
f(x_k + a*d_k) <= f(x_k) + c * a * grad_f(x_k)^T * d_k

5) now we can write an algorythm to search step (a)
a_k - start point, c_1 - const from (0, 0.5)
a = a_k
while (f(x_k + a*d_k) > f(x_k) + c * a * grad_f(x_k)^T * d_k) do
    a *= b (b - const and < 1)
end while

This way step can be found.
But there is problem: step may be too little.

About wolfe_condition:
lets add 1 more condition to armijo_condition:
0 < c_1 < c_2 < 1, c_1 = (0, 0.5)
armijo_cond
fi_k(a) <= fi_k(0) + a * c_1 * fi_k'(0)
+ second cond: |fi_k'(a)| <= |c_2 * fi_k'(0)|   

'''


def armijo_step(f, grad, x_k, a, b=0.5, c=0.1):
    d_k = -grad  # antigradient
    # grad_f(x_k)^T * d_k = np.dot(grad, d_k)
    # add a > 1e-8 to prevent infinite loop
    while f(x_k + a * d_k) > f(x_k) + c * a * np.dot(grad, d_k) and a > 1e-8:
        a *= b
    return a


def wolfe_step(f, grad, x_k, a=1, c_1=0.1, c_2=0.9, max_iter=20):
    f_x_k = f(x_k)
    d_k = -grad
    grad_x_k = np.dot(grad, d_k)  # grad_f(x_k)^T * d_k = np.dot(grad, d_k)
    left, right = 0, np.inf
    for i in range(max_iter):
        current_x = x_k + a * d_k
        current_f = f(current_x)
        current_grad = gradient(f, current_x)
        current_grad_k = np.dot(current_grad, d_k)  # grad_f(x_k)^T * d_k

        if current_f > f_x_k + c_1 * a * grad_x_k:  # armijo_cond
            # move right border
            right = a
            a = (left + right) / 2  # decreese a if f_new too large (binary search)
        elif abs(current_grad_k) > c_2 * abs(grad_x_k):  # second condition wolfe
            # step is too low -> increase step
            left = a
            if right == np.inf:
                a *= 2
            else:
                a = (left + right) / 2
        else:
            return (a, i)

    return a, i


def gradient_descent_armijo(f, x, step=1.0, max_iter=10000, eps=1e-6):
    for k in range(max_iter):
        grad = gradient(f, x)
        if np.linalg.norm(grad) < eps:
            break
        step = armijo_step(f, grad, x, step)
        x = x - step * grad
    print("gradient_descent_armijo iterations: ", k)
    return x


def gradient_descent_wolfe(f, x, step=1.0, max_iter=10000, eps=1e-6):
    for k in range(max_iter):
        grad = gradient(f, x)
        if np.linalg.norm(grad) < eps:
            break
        step, cnt = wolfe_step(f, grad, x, step)
        x = x - step * grad
    print("gradient_descent_wolfe: ", k, cnt)
    return x


def golden_section(f, a, b, x, grad, eps=1e-5):
    c = 2 / (1 + np.sqrt(5))

    x1 = b - (b - a) * c
    x2 = a + (b - a) * c

    # antigrad direction
    y1 = f(x - x1 * grad)
    y2 = f(x - x2 * grad)

    while abs(b - a) > eps:
        if y1 < y2:
            b, x2, y2 = x2, x1, y1
            x1 = b - (b - a) * c
            y1 = f(x - x1 * grad)
        else:
            a, x1, y1 = x1, x2, y2
            x2 = a + (b - a) * c
            y2 = f(x - x2 * grad)

    return (a + b) / 2


def gradient_descent_with_golden_section(f, x, a=0, b=0.1, max_iter=10000, eps=1e-6):
    for k in range(max_iter):
        grad = gradient(f, x)
        if np.linalg.norm(grad) < eps:
            break
        step = golden_section(f, a, b, x, grad)
        x = x - step * grad
    print("gradient_descent_with_golden_section iterations: ", k)
    return x


# dop task

# c > 2
def dichotomy_step(f, x, grad, a=0, b=1, eps=1e-5, c=10):
    while abs(b - a) > eps:
        # delta in (0, (b - a) / 2)
        delta = (b - a) / c
        x1 = (a + b) / 2 - delta
        x2 = (a + b) / 2 + delta

        y1 = f(x - x1 * grad)
        y2 = f(x - x2 * grad)

        if y1 > y2:
            a = x1
        else:
            b = x2

    return (a + b) / 2


def gradient_descent_dichotomy(f, x, a=0, b=0.1, max_iter=10000, eps=1e-6):
    for k in range(max_iter):
        grad = gradient(f, x)
        if np.linalg.norm(grad) < eps:
            break
        step = dichotomy_step(f, x, grad, a, b)
        x = x - step * grad
    print("gradient_descent_dichotomy iterations: ", k)
    return x


def print_result(method_name, result):
    print(f"{method_name}: \n \t \t \t [{result[0]:.16f}, {result[1]:.16f}]")
