import numpy as np
import sympy as sp

class Tracker:
    def __init__(self, g=0, h=0, f=0):
        self._g = g
        self._h = h
        self._f = f

    @property
    def g(self):
        return self._g

    @property
    def h(self):
        return self._h

    @property
    def f(self):
        return self._f

    @g.setter
    def g(self, value):
        self._g = value

    @h.setter
    def h(self, value):
        self._h = value

    @f.setter
    def f(self, value):
        self._f = value

def gradient(f, x, tracker, eps=1e-8):
    tracker.g = tracker.g + 1
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

    def wrapped(x, tracker):
        tracker.h = tracker.h + 1
        return np.array(hess_func(*x), dtype=float)

    return wrapped


def print_result(method_name, result):
    print(f"{method_name}: \t [{result[0]:.16f}, {result[1]:.16f}]")
