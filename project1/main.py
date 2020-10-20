import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class NearestNeighbour:
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def predict(self, x):
        return min(enumerate(self.y_data), key=lambda xi: np.sum((x - self.x_data[xi[0]]) ** 2))[1]


class MinErrorRate:
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data - 1
        self.n = x_data.shape[0]
        self.k = x_data.shape[1]
        self.c = len(set(y_data))
        self.weights = []
    def train(self):
        """
        Computes the weights for the discriminant function for class i

        :return: weights for the discriminant function of class i
        """

        for i in range(self.c):
            mu_i = np.atleast_2d(self.estimate_mu(i)).T
            cov_i = self.estimate_sigma(i)
            inv_sigma_i = np.linalg.inv(cov_i)
            P_i = self.estimate_P(i)
            print(P_i)
            Wi = -1 / 2 * inv_sigma_i
            wi = inv_sigma_i @ mu_i
            wi0 = -1 / 2 * mu_i.T @ inv_sigma_i @ mu_i - 1 / 2 * np.log(np.linalg.det(cov_i)) + np.log(P_i)
            self.weights.append([Wi, wi, wi0])
        return self.weights

    def estimate_P(self, i):
        return np.sum((y_train == i+1)) / len(y_train)

    def estimate_mu(self, i):
        return np.atleast_2d(np.mean(self.x_data[self.y_data == i], axis=0))

    def estimate_sigma(self, i):
        mu_i = self.estimate_mu(i)
        sigma_i = np.zeros((self.k, self.k))
        x = np.atleast_2d(self.x_data[self.y_data == i])
        print(x.shape)
        for xi in x:
            sigma_i += (xi - mu_i).T @ (xi - mu_i)
        return sigma_i / self.n

    def predict(self, x):
        ys = []
        x = np.atleast_2d(x).T
        for i in range(len(set(self.y_data))):
            weights = self.weights[i]
            ys.append(float(x.T @ weights[0] @ x + weights[1].T @ x + weights[2]))
        return np.argmax(np.array(ys))


class LeastSquares:
    def __init__(self, x_data, y_data):
        # append ones to the beginning of the training data
        ones = np.ones((x_data.shape[0], 1))
        self.x_data = np.hstack([ones, x_data])

        self.y_data = y_data

        self.b = np.where(self.y_data == 1, 1, -1)

        self.a = np.linalg.inv(self.x_data.T @ self.x_data) @ self.x_data.T @ self.b

    def predict(self, x):
        # insert 1 at the beginning of the vector
        x = np.insert(x, 0, 1)

        result = self.a.T @ x
        # Convert back to classes
        return np.where(result >= 0, 1, 2)



if __name__ == "__main__":
    data = pd.read_csv('data/ds-1.txt', sep="  ", header=None)
    data.columns = ["class", "feature_1", "feature_2", "feature_3", "feature_4"]
    print("Samples: ", len(data))
    print("Data Head:")
    print(data.head(5))
    print("Plotting Data")
    x = data.iloc[:, 1:].to_numpy()
    y = data.iloc[:, 0].to_numpy()
    x_train = x[1::2]
    y_train = y[1::2]
    x_test = x[::2]
    y_test = y[::2]
    # plt.scatter(x[:, 0], x[:, 1], c=y)
    # plt.show()
    # plt.scatter(x[:, 1], x[:, 2], c=y)
    # plt.show()
    nn_model = NearestNeighbour(x_train, y_train)
    me_model = MinErrorRate(x_train, y_train)
    me_model.train()
    ls_model = LeastSquares(x_train, y_train)

    print("Testing classifiers...")

    # Nearest Neighbour
    c = 0
    for i in range(len(x_test)):
        if nn_model.predict(x_test[i]) == y_test[i]:
            c += 1
    print(f"Nearest Neighbour accuracy: {c / len(x_test):.2%}")

    # MinErrorRate
    c = 0
    for i in range(len(x_test)):
        if me_model.predict(x_test[i]) == y_test[i] - 1:
            c += 1
    print(f"Min Error accuracy: {c / len(x_test):.2%}")

    # Least Squares
    c = 0
    for i in range(len(x_test)):
        if ls_model.predict(x_test[i]) == y_test[i]:
            c += 1
    print(f"Least Squares accuracy: {c / len(x_test):.2%}")

    print("Tests complete")
