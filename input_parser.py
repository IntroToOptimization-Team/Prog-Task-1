import numpy as np


def read_input(filepath):
    with open(filepath, mode="rt") as file:
        C = np.array(list(map(float, file.readline().split())))
        file.readline()

        A = []
        while True:
            line = file.readline()

            if line.strip() == "":
                break

            A.append(np.array(list(map(float, line.split()))))
        A = np.array(A)
        
        b = np.array(list(map(float, file.readline().split())))
        file.readline()

        eps = float(file.readline().strip())
        file.readline()
        return {"C": C, "A": A, "b": b, "eps": eps}
