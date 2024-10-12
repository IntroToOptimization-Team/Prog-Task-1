import os

from input_parser import read_input
import numpy as np


def simplex(C, A, b, eps=0.001):
    # Step 1. Print the optimization problem
    print("Problem:")
    print("max: z = ", end="")
    print(" + ".join([f"{C[i]} * x{i + 1}" for i in range(len(C))]))
    print()

    print("subject to the constraints:")
    for row in range(len(b)):
        print(' + '.join([f'{A[row][i]} * x{i + 1}' for i in range(len(A[row]))]) + ' <= ' + str(b[row]))
    print()

    # Step 2. Initialize
    C = np.concatenate((C, np.zeros(b.shape[0])))

    tableau = np.concatenate((A, np.eye(b.shape[0])), axis=1)
    # current_solution = np.zeros(b.shape[0])
    variable_indexes = list(range(A.shape[1], len(C)))

    # z = np.array([0 for _ in range(len(C))])
    objective_row = -C

    solz = 0

    # Step 3. Iteratively apply the Simplex method
    while True:
        if all(objective_row >= 0):
            # Solution optimal
            solution = np.zeros((tableau.shape[1]))
            for i in range(len(variable_indexes)):
                solution[variable_indexes[i]] = b[i]

            solution = solution[:A.shape[1]]
            return {
                'solver_state': 'solved',
                'x*': list(map(lambda x: round(float(x), 6), solution)),
                # 'z': float(current_solution.T @ b)
                'z': solution.T @ C[:A.shape[1]]
            }

        # Identify the entering variable
        entering_variable_index = np.argmin(objective_row)

        # Identify the leaving variable
        leaving_variable_index = -1
        for index in range(len(b)):
            if tableau[index][entering_variable_index] != 0 and b[index] / tableau[index][entering_variable_index] > 0:
                if leaving_variable_index == -1:
                    leaving_variable_index = index
                else:
                    leaving_variable_index = leaving_variable_index if b[leaving_variable_index] / tableau[
                        leaving_variable_index, entering_variable_index] < b[index] / tableau[
                                                                           index, entering_variable_index] else index

        if leaving_variable_index == -1:
            # problem is unbounded
            return {'solver_state': 'unbounded'}

        # Perform pivot operations
        # leaving_variable_index - row
        # entering_variable_index - column

        variable_indexes[leaving_variable_index] = entering_variable_index
        # current_solution[leaving_variable_index] = C[entering_variable_index]

        b[leaving_variable_index] /= tableau[leaving_variable_index, entering_variable_index]
        tableau[leaving_variable_index] /= tableau[leaving_variable_index, entering_variable_index]

        for row in range(A.shape[0]):
            if row == leaving_variable_index:
                continue

            b[row] -= b[leaving_variable_index] * tableau[row][entering_variable_index]
            tableau[row] -= tableau[leaving_variable_index] * tableau[row][entering_variable_index]

        dsol = objective_row[entering_variable_index] * b[leaving_variable_index]
        objective_row -= tableau[leaving_variable_index] * objective_row[entering_variable_index]
        if -dsol < eps:
            solution = np.zeros((tableau.shape[1]))
            for i in range(len(variable_indexes)):
                solution[variable_indexes[i]] = b[i]

            solution = solution[:A.shape[1]]
            return {
                'solver_state': 'solved',
                'x*': list(map(lambda x: round(float(x), 6), solution)),
                'z': solution.T @ C[:A.shape[1]]
            }


if __name__ == "__main__":
    for file in os.listdir("tests"):
        res = simplex(**read_input(f"tests/{file}"))
        if res["solver_state"] == "unbounded":
            print("The method is not applicable!")
        else:
            print("The solution:")
            print(" ".join(map(lambda x: str(round(x, 6)), res["x*"])))
            print(round(res["z"], 6))
        print("-" * 40)
