import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from main import rosenbrock, himmelblau, f, x, y, rosenbrock_sympy, himmelblau_sympy, f_sympy
from methods import scipy_lbfgs, scipy_bfgs, scipy_newton_cg, bfgs_method, newton_method_wolfe_step, \
    newton_method_golden_section

if not os.path.exists('plots'):
    os.makedirs('plots')


def func_np(func):
    def wrapper(x, y):
        return np.array([func(np.array([xi, yi])) for xi, yi in zip(x.flatten(), y.flatten())]).reshape(x.shape)

    return wrapper


def scipy_method_track(method_func, f, x0, max_iter=100, eps=1e-6):
    x = x0.copy()
    trajectory = [x.copy()]

    def callback(xk):
        trajectory.append(xk.copy())

    result = method_func(f, x0, eps=eps, max_iter=max_iter, callback=callback)

    return np.array(trajectory)


def plot_2d(func, trajectories, names, title, filename, x_range=(-5, 5), y_range=(-5, 5)):
    rosenbrock_np = func_np(func)

    x_grid = np.linspace(x_range[0], x_range[1], 300)
    y_grid = np.linspace(y_range[0], y_range[1], 300)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = rosenbrock_np(X, Y)

    plt.figure(figsize=(12, 10))

    contour = plt.contour(X, Y, Z, 50, colors='blue', alpha=0.5)
    plt.contourf(X, Y, Z, 50, cmap='viridis', alpha=0.5)
    plt.colorbar(label='Function Value')

    colors = ['red', 'orange', 'magenta', 'cyan', 'yellow', 'green']
    for i, (traj, name) in enumerate(zip(trajectories, names)):
        plt.plot(traj[:, 0], traj[:, 1], 'o-', color=colors[i % len(colors)], label=name, linewidth=2, markersize=4)
        plt.plot(traj[-1, 0], traj[-1, 1], 'x', color=colors[i % len(colors)], markersize=10)

    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.legend()

    plt.savefig(f'plots/{filename}_2d.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_3d(func, trajectories, names, title, filename, x_range=(-5, 5), y_range=(-5, 5)):
    rosenbrock_np = func_np(func)

    x_grid = np.linspace(x_range[0], x_range[1], 100)
    y_grid = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = rosenbrock_np(X, Y)

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.8, linewidth=0, antialiased=True)

    colors = ['red', 'orange', 'magenta', 'cyan', 'yellow', 'green']

    for i, (traj, name) in enumerate(zip(trajectories, names)):
        z_vals = np.array([func(point) for point in traj])
        ax.plot(traj[:, 0], traj[:, 1], z_vals, 'o-', color=colors[i % len(colors)],
                label=name, linewidth=2, markersize=4)
        ax.plot([traj[-1, 0]], [traj[-1, 1]], [func(traj[-1])], 'x', color=colors[i % len(colors)], markersize=10)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    ax.set_title(title)
    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Function Value')
    ax.legend()

    plt.savefig(f'plots/{filename}_3d.png', dpi=300, bbox_inches='tight')
    plt.close()


def run_visualization(func, func_sympy, func_name, start_point, x_range=(-5, 5), y_range=(-5, 5)):
    print(f"Generating visualizations for {func_name} function...")

    _, newton_golden_traj = newton_method_golden_section(
        func, func_sympy, variables=[x, y], x0=start_point, max_iter=100)

    _, newton_wolfe_traj = newton_method_wolfe_step(
        func, func_sympy, variables=[x, y], x0=start_point, max_iter=100)

    _, bfgs_traj = bfgs_method(func, x0=start_point, max_iter=100)

    scipy_newton_traj = scipy_method_track(scipy_newton_cg, func, start_point)
    scipy_bfgs_traj = scipy_method_track(scipy_bfgs, func, start_point)
    scipy_lbfgs_traj = scipy_method_track(scipy_lbfgs, func, start_point)

    trajectories = [
        newton_golden_traj,
        newton_wolfe_traj,
        bfgs_traj,
        scipy_newton_traj,
        scipy_bfgs_traj,
        scipy_lbfgs_traj
    ]

    names = [
        'Newton (Golden)',
        'Newton (Wolfe)',
        'BFGS',
        'Newton-CG (scipy)',
        'BFGS (scipy)',
        'L-BFGS-B (scipy)'
    ]

    plot_2d(func, trajectories, names,
            f'Optimization Paths for {func_name} Function',
            f'{func_name.lower()}_optimization', x_range, y_range)

    plot_3d(func, trajectories, names,
            f'3D Optimization Paths for {func_name} Function',
            f'{func_name.lower()}_optimization', x_range, y_range)

    print(f"Visualizations for {func_name} function completed.")


def main():
    rosenbrock_start = np.array([1.5, 1.5])
    himmelblau_start = np.array([-4.0, 4.0])
    quadratic_start = np.array([15.0, -15.0])

    run_visualization(
        rosenbrock, rosenbrock_sympy, 'Rosenbrock',
        rosenbrock_start, x_range=(-2, 2), y_range=(-1, 3)
    )

    run_visualization(
        himmelblau, himmelblau_sympy, 'Himmelblau',
        himmelblau_start, x_range=(-6, 6), y_range=(-6, 6)
    )

    run_visualization(
        f, f_sympy, 'Quadratic',
        quadratic_start, x_range=(-20, 20), y_range=(-20, 20)
    )


if __name__ == "__main__":
    main()
