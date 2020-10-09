import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class NearestNeighbour:
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def predict(self, x):
        return min(enumerate(self.y_data), key=lambda xi: np.sum((x-self.x_data[xi[0]])**2))[1]


class MinErrorRate:
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
        self.n = x_data.shape[0]
        self.k = x_data.shape[1]
        self.c = len(set(y_data))

        self.train(1)

    def train(self, i):
        """
        Computes the weights for the discriminant function for class i

        :param i: class for which to parametrize
        :return: weights for the discriminant function of class i
        """
        mu = self.estimate_mu()
        cov = self.estimate_sigma()
        inv_sigmas = np.array([np.linalg.inv(cov[:,:,i]) for i in range(self.c)])
        P = self.estimate_P()
        Wi = -1/2 * inv_sigmas[:,:,i]
        print(mu.shape)
        inv_s_i = inv_sigmas[1,:,:]
        print(inv_s_i.shape)
        wi = inv_s_i @ np.atleast_2d(mu).T
        wi0 = -1/2 * np.atleast_2d(mu) @ inv_s_i @ np.atleast_2d(mu).T - 1/2 * np.abs(cov[i,:,:])+np.log(P)
        return [Wi, wi, wi0]

    def estimate_P(self):
        P = np.zeros(len(set(self.y_data)))
        for c in range(len(set(self.y_data))):
            P[c] = np.sum((y_train == c+1))/len(y_train)
        return P

    def estimate_mu(self):
        return np.mean(self.x_data, axis=0)

    def estimate_sigma(self):
        sigma = np.zeros((self.k, self.k, self.c))
        for i in range(self.c):
            mu = np.mean(self.x_data, axis=0)
            sigma_i = np.zeros((self.k, self.k))
            for k in range(self.n):
                sigma_i+=np.atleast_2d(self.x_data[i]-mu).T@np.atleast_2d(self.x_data[i]-mu)
            sigma[:,:,i]=sigma_i
        return sigma/self.n


    def predict(self, x):
        weights = []
        for classes in range(len(set(self.y_data))):
            weights.append(self.train(classes))






class LeastSquares:
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def predict(self, x):
        pass


if __name__ == "__main__":
    data = pd.read_csv('data/ds-1.txt', sep="  ", header=None)
    data.columns = ["class", "feature_1", "feature_2", "feature_3", "feature_4"]
    print("Samples: ", len(data))
    print("Data Head:")
    print(data.head(5))
    print("Plotting Data")
    x = data.iloc[:, 1:].to_numpy()
    y = data.iloc[:, 0].to_numpy()
    x_train = x[1:][::2]
    y_train = y[1:][::2]
    x_test = x[::2]
    y_test = y[::2]
    # plt.scatter(x[:, 0], x[:, 1], c=y)
    # plt.show()
    # plt.scatter(x[:, 1], x[:, 2], c=y)
    # plt.show()
    nn_model = NearestNeighbour(x,y)
    me_model = MinErrorRate(x,y)

    print("Testing classifiers...")
    for i in range(len(x_train)):
        assert nn_model.predict(x_train[i]) == y_train[i], "Nearest Neigbour Classifier not predicting correctly"
    #MinErrorRate

    print("Tests complete")
