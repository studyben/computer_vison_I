"""
Linear Regression model
"""

import numpy as np


class Linear(object):
    def __init__(self, n_class: int, lr: float, epochs: int, weight_decay: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # Initialize in train
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class
        self.weight_decay = weight_decay

    def train(self, X_train: np.ndarray, y_train: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Train the classifier.

        Use the linear regression update rule as introduced in the Lecture.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """

        N, D = X_train.shape
        self.w = weights
        #ones = np.ones((N, 1))
        #X_trains = np.hstack((X_train, ones))
        X_trains = X_train
        #print("Xtrain shape: ", np.shape(X_trains))
        #print("YTrain shape: ", np.shape(y_train))
        #print("w shape: ", np.shape(self.w))
        
        for i in range(self.epochs):
            for j in range(self.n_class):
                indices = np.where(y_train == j)
                data = X_trains[indices, :] 
                w = self.w[j, :]
                y_hat = np.where(y_train == j, 1, -1)
                # dy/dw = 1/n * sum(x * (y_hat - x_train * w)) + weight_decay * w * nrom(w)
                dydw = 1/N * np.transpose(X_trains) @ (y_hat - X_trains @ w) +\
                             self.weight_decay * w * np.linalg.norm(w, ord = 1)
                self.w[j, :] = w + self.lr * dydw
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
        #ones = np.ones((N, 1))
        #X_tests = np.hstack((X_test, ones))
        X_tests = X_test
        results = np.zeros((N, self.n_class))
        
        for i in range(self.n_class):
            results[:, i] = X_tests @ self.w[i, :]
        top = np.argmax(results, axis=1)
        
        return top
        # TODO: implement me