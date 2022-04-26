"""
K Nearest Neighbours Model
"""
import numpy as np
from statistics import mode

from statistics import mode

class KNN(object):
    def __init__(self, num_class: int):
        self.num_class = num_class

    def train(self, x_train: np.ndarray, y_train: np.ndarray, k: int):
        """
        Train KNN Classifier

        KNN only need to remember training set during training

        Parameters:
            x_train: Training samples ; np.ndarray with shape (N, D)
            y_train: Training labels  ; snp.ndarray with shape (N,)
        """
        self._x_train = x_train
        self._y_train = y_train
        self.k = k

    def predict(self, x_test: np.ndarray, k: int = None, loop_count: int = 1):
        """
        Use the contained training set to predict labels for test samples

        Parameters:
            x_test    : Test samples                                     ; np.ndarray with shape (N, D)
            k         : k to overwrite the one specificed during training; int
            loop_count: parameter to choose different knn implementation ; int

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # Fill this function in
        k_test = k if k is not None else self.k

        if loop_count == 1:
            distance = self.calc_dis_one_loop(x_test)
        elif loop_count == 2:
            distance = self.calc_dis_two_loop(x_test)

        #print("shape of distance: ", np.shape(distance))
        #print("Distance type:", type(distance))
        length, _ = np.shape(x_test)
        result = np.zeros(length)
        for i in range(length):
            dline = distance[i]
            #print("dline shape: ", np.shape(dline))
            c = np.argpartition(dline, k_test)
            #print("shape of c: ", np.shape(c))
            x = c.reshape(-1)
            x = x[:k_test]
            #print("shape of x: ", np.shape(x))
            #print(x)
            #print(self._y_train[x])
            result[i] = mode(self._y_train[x])
        return result

    def calc_dis_one_loop(self, x_test: np.ndarray):
        """
        Calculate distance between training samples and test samples

        This function could one for loop

        Parameters:
            x_test: Test samples; np.ndarray with shape (N, D)
        """
        distances = []
        # TODO: implement me
        leng, _ = np.shape(self._x_train)
        for i in range(leng):
            distance = ((self._x_train[i] - x_test)**2)
            distance = distance.sum(axis=1)
            distance = np.sqrt(distance)
            distances.append(distance)
        distances = np.array(distances)
        distances = distances.T
        return distances

    def calc_dis_two_loop(self, x_test: np.ndarray):
        """
        Calculate distance between training samples and test samples

        This function could contain two loop

        Parameters:
            x_test: Test samples; np.ndarray with shape (N, D)
        """
        # TODO: implement me
        distances = []
        for i in range(len(x_test)):
            dt = []
            for j in range(len(self._x_train)):
                distance = ((self._x_train[j] - x_test[i]) ** 2)
                distance = distance.sum()
                distance = np.sqrt(distance)
                dt.append(distance)
            distances.append(dt)
        return distances