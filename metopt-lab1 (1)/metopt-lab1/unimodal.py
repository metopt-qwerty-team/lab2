from methods import *

x0_f1 = np.array([15.0, -15.0])
x0_f2 = np.array([15.0, -15.0])
x0_f3 = np.array([15.0, -15.0])

f1 = lambda x: x[0] ** 2 + x[1] ** 2
f2 = lambda x: 1000 * (x[0] ** 2) + x[1] ** 2

print("f1(x, y) = x^2 + y^2:")
print("good: start point (15, -15)")
print_result("Gradient Descent (Constant Step) (step = 0.1, max_iterations = 1000) value of iterations: 79",
             gradient_descent_with_constant_step(f1, x0_f1, 0.1, 1000))
print_result("Gradient Descent (Decreasing Step) (step = 1, max_iterations = 1000) value of iterations: 3",
             gradient_descent_with_decreasing_step(f1, x0_f1, 1, 1000))
print_result("Gradient Descent (Armijo) (step = 0.1, max_iterations = 1000) value of iterations: 79",
             gradient_descent_armijo(f1, x0_f1, 0.1, 1000))
print_result("Gradient Descent (Wolfe) (step = 0.001, max_iterations = 1000) value of iterations: 129",
             gradient_descent_wolfe(f1, x0_f1, 0.001, 1000))
print("примечание к работе Wolfe. Он хорошо отрабатывает на плохом случае для Armijo, "
      "потому что не позволяет шагу быть слишком маленьким")
print_result("Gradient Descent (Golden Section) (step from (0, 0.1), max_iterations = 1000) value of iterations: 79",
             gradient_descent_with_golden_section(f1, x0_f1, 0, 0.1, 1000))
print_result("Gradient Descent (Dichotomy) (step from (0, 0.1), max_iterations = 1000) value of iterations: 101",
             gradient_descent_dichotomy(f1, x0_f1, 0, 0.08, 1000))
print("-----------------------------------------------")

print("bad: start point (15, -15)")
print_result("Gradient Descent (Constant Step) (step = 1, max_iterations = 1000) value of iterations: 999",
             gradient_descent_with_constant_step(f1, x0_f1, 1, 1000))
print("плохо, потому что из-за подобранного шага на каждой итерации происходит "
      "перескок на ту же линию уровня, на которой находились изначально")
print()
print_result("Gradient Descent (Decreasing Step) (step = 0.1, max_iterations = 10000) value of iterations: 9999",
             gradient_descent_with_decreasing_step(f1, x0_f1, 0.1, 10000))
print("плохо, потому что изначально находимся далеко от минимума, берем маленький шаг, и из-за его уменьшения"
      "на каждой итерации просто не можем добраться до минимума")
print()
print_result("Gradient Descent (Armijo) (step = 0.001, max_iterations = 1000) value of iterations: 999",
             gradient_descent_armijo(f1, x0_f1, 0.001, 1000))
print("плохо, потому что из-за маленького шага не хватает итераций для вычисления минимума")
print()
print_result("Gradient Descent (Wolfe) (step = 0.001, max_iterations = 10) value of iterations: 9",
             gradient_descent_wolfe(f1, x0_f1, 0.001, 10))
print("плохо, потому что не хватает числа итераций для поиска минимума")
print()
print_result("Gradient Descent (Golden Section) (step from (0, 0.1), max_iterations = 10) value of iterations: 9",
             gradient_descent_with_golden_section(f1, x0_f1, 0, 0.1, 10))
print("плохо, потому что не хватает числа итераций для поиска минимума")
print()
print_result("Gradient Descent (Dichotomy) (step from (0, 0.1), max_iterations = 1000) value of iterations: 9",
             gradient_descent_dichotomy(f1, x0_f1, 0, 0.08, 10))
print("плохо, потому что не хватает числа итераций для поиска минимума")
print()
print("-----------------------------------------------")
print("-----------------------------------------------")
print()
print("f2(x, y) = 1000*x^2 + y^2:")
print("good: start point (15, -15)")
print_result("Gradient Descent (Constant Step) (step = 0.0001, max_iterations = 100000) value of iterations: 86075",
             gradient_descent_with_constant_step(f2, x0_f2, 0.0001, 100000))
print("Decreasing Step для этой функции не подобрать работающий")
# print_result("Gradient Descent (Decreasing Step) (step = 0.1, max_iterations = 10000) value of iterations: 9999",
#   gradient_descent_with_decreasing_step(f2, x0_f2, 0.0001, 10000))
print_result("Gradient Descent (Armijo) (step = 0.1, max_iterations = 100000) value of iterations: 11011",
             gradient_descent_armijo(f2, x0_f2, 0.1, 100000))
print_result("Gradient Descent (Wolfe) (step = 1, max_iterations = 10000) value of iterations: 8429",
             gradient_descent_wolfe(f2, x0_f2, 1, 10000))
print_result("Gradient Descent (Golden Section) (step from (0, 0.1), max_iterations = 10000) value of iterations: 7000",
             gradient_descent_with_golden_section(f2, x0_f2, 0, 0.1, 10000))
print_result("Gradient Descent (Dichotomy) (step from (0, 0.1), max_iterations = 10000) value of iterations: 201",
             gradient_descent_dichotomy(f2, x0_f2, 0, 0.1, 10000))
print("-----------------------------------------------")

print("bad: start point (15, -15)")
print_result("Gradient Descent (Constant Step) (step = 0.001, max_iterations = 1000) value of iterations: 999",
             gradient_descent_with_constant_step(f2, x0_f2, 0.001, 100000))
print(
    "плохо, потому что из-за плохой обусловленности функции (1000*x^2 vs y^2) и слишком большого шага метод расходится")
print()
print_result("Gradient Descent (Decreasing Step) (step = 1.0, max_iterations = 100) value of iterations: 99",
             gradient_descent_with_decreasing_step(f2, x0_f2, 0.01, 10000))
print(
    "плохо, потому что начальный шаг слишком большой, что приводит к расходимости, а уменьшение шага не успевает компенсировать это за малое число итераций")
print()
print_result("Gradient Descent (Armijo) (step = 0.1, max_iterations = 1000) value of iterations: 999",
             gradient_descent_armijo(f2, x0_f2, 0.1, 1000))
print("плохо, потому что не хватает итераций для сходимости из-за плохой обусловленности функции")
print()
print_result("Gradient Descent (Wolfe) (step = 0.1, max_iterations = 1000) value of iterations: 999",
             gradient_descent_wolfe(f2, x0_f2, 0.1, 1000))
print("плохо, потому что не хватает числа итераций для поиска минимума из-за плохой обусловленности")
print()
print_result("Gradient Descent (Golden Section) (step from (0, 0.1), max_iterations = 1000) value of iterations: 999",
             gradient_descent_with_golden_section(f2, x0_f2, 0, 0.1, 1000))
print("плохо, потому что не хватает числа итераций для поиска минимума")
print()
print_result("Gradient Descent (Dichotomy) (step from (0, 1), max_iterations = 100) value of iterations: 99",
             gradient_descent_dichotomy(f2, x0_f2, 0, 1.0, 100))
print("плохо, потому что не хватает числа итераций для поиска минимума")
print()
print("-----------------------------------------------")
print("-----------------------------------------------")
print()
