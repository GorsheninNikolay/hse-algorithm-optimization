import numpy as np
import simplex
import math


def branch_and_bound(c, A, b):
    best_solution, best_value = None, -np.inf

    active_nodes = [(A, b)]

    while active_nodes:
        current_A, current_b = active_nodes.pop(0)

        wizards, knights, dragons, value = simplex.simplex_method(
            c, current_A, current_b
        )
        solution = np.array([wizards, knights, dragons])

        if value <= best_value:
            continue

        if np.all(np.floor(solution) == solution):
            if value > best_value:
                best_value = value
                best_solution = solution
        else:
            branch_var = np.argmax(solution - np.floor(solution))

            new_A1 = np.vstack([current_A, np.eye(1, len(c), branch_var)])
            new_b1 = np.append(current_b, math.floor(solution[branch_var]))

            new_A2 = np.vstack([current_A, np.eye(1, len(c), branch_var)])
            new_b2 = np.append(current_b, -math.ceil(solution[branch_var]))

            active_nodes.append((new_A1, new_b1))
            active_nodes.append((new_A2, new_b2))

    return best_solution, best_value


if __name__ == "__main__":
    m1 = 2
    m2 = 2
    m3 = 1

    k1 = 3
    k2 = 1
    k3 = 0

    d1 = 4
    d2 = 3
    d3 = 1

    A = [
        [m1, k1, d1],
        [m2, k2, d2],
        [m3, k3, d3],
    ]

    R = [12, 5, 2]  # Запасы
    P = [5, 7, 8]  # Сила

    best_solution, best_value = branch_and_bound(P, A, R)
    wizards, knights, dragons = best_solution
    print(
        f"Маги={wizards}, Рыцарей={knights}, Драконов={dragons}, целевое значение={best_value}"
    )
