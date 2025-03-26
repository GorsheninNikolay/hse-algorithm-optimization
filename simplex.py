import numpy as np


def simplex_method(z_to_max, coefficients, limitations):
    z_to_max = [-x for x in z_to_max]  # to maximize target value

    # init simplex table
    num_constraints, num_vars = len(limitations), len(z_to_max)
    slack_vars = np.eye(num_constraints)
    A = np.hstack([coefficients, slack_vars])
    c = np.hstack([z_to_max, np.zeros(num_constraints)])

    table = np.zeros((num_constraints + 1, num_vars + num_constraints + 1))
    table[:-1, : num_vars + num_constraints] = A
    table[:-1, -1] = limitations
    table[-1, : num_vars + num_constraints] = c
    table[-1, -1] = 0  # Значение целевой функции

    # print(f"Simplex table:\n{table}\n")

    while True:
        # Step 1: Find the input line
        pivot_col = np.argmin(table[-1, :-1])
        if table[-1, pivot_col] >= 0:
            break

        # Step 2: Find the output line
        ratios = np.inf * np.ones(num_constraints)
        for i in range(num_constraints):
            if table[i, pivot_col] > 0:
                ratios[i] = table[i, -1] / table[i, pivot_col]
        pivot_row = np.argmin(ratios)

        if ratios[pivot_row] == np.inf:
            raise ValueError("The task is unlimited")

        # Step 3: Normalize pivot-line
        pivot_val = table[pivot_row, pivot_col]
        table[pivot_row, :] /= pivot_val

        # Step 4: Update other lines
        for i in range(num_constraints + 1):
            if i != pivot_row:
                multiplier = table[i, pivot_col]
                table[i, :] -= multiplier * table[pivot_row, :]

        # print(f"Итерация. Pivot: строка {pivot_row}, столбец {pivot_col}")
        # print(table)
        # print("\n")

    # Извлекаем решение
    solution = np.zeros(num_vars)
    for i in range(num_vars):
        col = table[:, i]
        if (
            np.sum(np.abs(col - 1) < 1e-8) == 1
            and np.sum(np.abs(col) < 1e-8) == num_constraints
        ):
            solution[i] = table[np.where(np.abs(col - 1) < 1e-8)[0][0], -1]

    wizards, knights, dragons = solution[:num_vars]

    return wizards, knights, dragons, table[-1, -1]


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

    R = [20, 20, 20]  # Запасы
    P = [5, 7, 8]  # Сила

    wizards, knights, dragons, target_val = simplex_method(P, A, R)
    print(
        f"Маги={wizards}, Рыцарей={knights}, Драконов={dragons}, целевое значение={target_val}"
    )
