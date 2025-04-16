from methods import *

f3 = lambda x: (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2
start_points = [[-4.0, -4.0], [-4.0, 4.0], [2.0, -4.0], [4.0, -2.0]]
for point in start_points:
    print_result(f"Gradient Descent (Dichotomy) for func-himmelblau and point: {point}",
                 gradient_descent_dichotomy(f3, point, 0, 1, 1000))
print(
    "У функции Химмельблау известно 4 минимума. Мы привели 4 точки начала, с которых происходит сваливание в каждую из них.")

print()
print("-----------------------------------------------")
print("-----------------------------------------------")
print()
