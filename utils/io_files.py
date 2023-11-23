import numpy as np
import json
import os
import shutil


class JsonConfig_stat:
        def __init__(self, json_path):
            with open(json_path) as f:
                self.config_dict = json.load(f)

            self.dir_path = os.path.join(os.getcwd(), self.config_dict.get("directory"))
            self.dir_path_output = os.path.join(self.dir_path, "output")
            self.dir_path_images = os.path.join(self.dir_path, "images")
            self.dir_path_slices = os.path.join(self.dir_path_images, "slices")
            self.dir_path_cubes = os.path.join(self.dir_path_images, "cubes")

            self.orientation = np.array(self.config_dict.get("orientation"))
            self.orientation = self.orientation / np.sqrt(self.orientation.dot(self.orientation))
            self.center_point = np.array(self.config_dict.get("center_point"))
            self.L = self.config_dict.get("L")
            self.hwl_lenghts = np.array([self.L, self.L, self.L])
            self.n_x = self.config_dict.get("N")
            self.n_discrete_hwl = np.array([self.n_x, self.n_x, self.n_x])
            self.k = self.config_dict.get("k")
            self.E0 = self.config_dict.get("E0")
            self.exp_no = self.config_dict.get("experiment_number")
            self.seed = self.config_dict.get("seed")

            self.__create_dir()

        def __create_dir(self):
            if os.path.isdir(self.dir_path):
                shutil.rmtree(self.dir_path)
            os.mkdir(self.dir_path)
            os.mkdir(self.dir_path_output)
            os.mkdir(self.dir_path_images)
            os.mkdir(self.dir_path_slices)
            os.mkdir(self.dir_path_cubes)
            return True

        def save_file_results(self, U_vec, iterations = None):
            with open(os.path.join(self.dir_path_output, f"result_info_{self.exp_no}.txt"), "w") as file:
                file.write("Resulting discretization:\n")
                file.write(f"{self.n_x}\t{self.n_x}\t{self.n_x}\n")
                file.write(f"Total fragments:{self.n_x ** 3}\n\n")
                file.write(f"Orientation of wave is: d_vec = ({self.orientation[0]}, {self.orientation[1]}, {self.orientation[2]})\n\n")
                file.write(f"Wave number is: k = {self.k}\n\n")
                file.write(f"Linear size of cube: L = {self.L}\n\n")
                file.write(f"Amplitude is: E0 = {self.E0}\n\n")
                if iterations is not None:
                    file.write(f"Total number of TwoSGD iterations is: {iterations}")

            save_np_file_txt(array_numpy=np.real(U_vec), filename=os.path.join(self.dir_path_output,
                                                                      f"resulting_real_vector_{self.exp_no}.txt"))
            save_np_file_txt(array_numpy=np.imag(U_vec), filename=os.path.join(self.dir_path_output,
                                                                      f"resulting_imag_vector_{self.exp_no}.txt"))
            return True


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