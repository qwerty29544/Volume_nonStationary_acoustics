import numpy as np


def save_np_file_txt(array_numpy, filename):
    with open(filename, "w") as file:
        file.write(" ".join(list(map(str, array_numpy.shape))))
        file.write("\n")
        file.write("\t".join(list(map(str, array_numpy.reshape(-1)))))
    return 0


def load_np_file_txt(filename, array_type=complex):
    with open(filename, "r") as file:
        shape = tuple(map(int, file.readline().strip().split(" ")))
        array = np.array(file.readline().strip().split("\t"),
                         dtype=array_type).reshape(shape)
    return array


if __name__=="__main__":
    array_arr = np.random.uniform(0, 1, 5**3).reshape((5, 5, 5))
    save_np_file_txt(array_arr, filename="array_test.txt")
    array_test = load_np_file_txt(filename="array_test.txt", array_type=float)
    print(np.all(array_arr == array_test))