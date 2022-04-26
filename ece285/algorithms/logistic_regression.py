"""
Logistic regression model
"""

import numpy as np
import math


class Logistic(object):
    def __init__(self, n_class: int, lr: float, epochs: int, weight_decay: float):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class
        self.threshold = 0.5  # To threshold the sigmoid
        self.weight_decay = weight_decay

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        return 1 / (1 + np.exp(-z))
        # TODO: implement me
        pass

    def train(self, X_train: np.ndarray, y_train: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.
        Train a logistic regression classifier for each class i to predict the probability that y=i

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """

        N, D = X_train.shape
        self.w = weights
        # ones = np.ones((N, 1))
        # X_trains = np.hstack((X_train, ones))
        X_trains = X_train
        # print("Xtrain shape: ", np.shape(X_trains))
        # print("YTrain shape: ", np.shape(y_train))
        # print("w shape: ", np.shape(self.w))

        for i in range(self.epochs):
            for j in range(self.n_class):
                indices = np.where(y_train == j)
                w = self.w[j, :]
                y_hat = np.where(y_train == j, 1, -1)
                y_hat_diag = np.diag(y_hat)
                # dy/dw = 1/n * sum(x * (y_hat - x_train * w)) + weight_decay * w * nrom(w)
                p1 = np.diag(self.sigmoid(-y_hat_diag @ (X_trains @ w)))
                dydw = -1 / N * np.transpose(X_trains) @ (p1 @ y_hat) + self.weight_decay * w
                self.w[j, :] = w - self.lr * dydw
        # TODO: implement me

        return self.w

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        N, _ = X_test.shape
        # ones = np.ones((N, 1))
        # X_tests = np.hstack((X_test, ones))
        X_tests = X_test
        results = np.zeros((N, self.n_class))

        for i in range(self.n_class):
            results[:, i] = self.sigmoid(X_tests @ self.w[i, :])
        top = np.argmax(results, axis=1)

        return top
        # TODO: implement me
        pass