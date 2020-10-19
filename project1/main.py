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
        self.y_data = y_data-1
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
        mu_i = np.atleast_2d(self.estimate_mu(i)).T
        cov_i = self.estimate_sigma(i)
        inv_sigma_i =np.linalg.inv(cov_i)
        P_i = self.estimate_P(i)
        Wi = -1/2 * inv_sigma_i
        wi = inv_sigma_i @ mu_i
        wi0 = -1/2 * mu_i.T @ inv_sigma_i @ mu_i - 1/2 * np.linalg.norm(cov_i)+np.log(P_i)
        return [Wi, wi, wi0]

    def estimate_P(self, i):
        P = np.zeros(len(set(self.y_data)))
        for c in range(len(set(self.y_data))):
            P[c] = np.sum((y_train == c+1))/len(y_train)
        return P[i]

    def estimate_mu(self, i):
        return np.mean(self.x_data[self.y_data == i], axis=0)


    def estimate_sigma(self, i):
        mu_i = self.estimate_mu(i)
        sigma_i = np.zeros((self.k, self.k))
        x = self.x_data[self.y_data==i]
        for k in range(self.n):
            sigma_i+=np.atleast_2d(x[i]-mu_i).T@np.atleast_2d(x[i]-mu_i)
        return sigma_i/len(x)


    def predict(self, x):
        ys = []
        x = np.atleast_2d(x).T
        for i in range(len(set(self.y_data))):
            weights = self.train(i)
            ys.append(float(x.T@weights[0]@x+weights[1].T@x+weights[2]))
        return np.argmax(np.array(ys))



class LeastSquares:
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def predict(self, x):
        extra = 0
        dim = len(set(self.y_data))+extra
        a = np.zeros((1, dim))



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
    nn_model = NearestNeighbour(x_train,y_train)
    me_model = MinErrorRate(x_train,y_train)

    print("Testing classifiers...")
    for i in range(len(x_train)):
        assert nn_model.predict(x_train[i]) == y_train[i], "Nearest Neigbour Classifier not predicting correctly"
    #MinErrorRate

    print("Evaluating classifiers...")
    c=0
    for i in range(len(x_test)):
        if me_model.predict(x_test[i])==y_test[i]-1:
            c+=1
    print("Minimum Error Rate classifier Accuracy: ", c/len(x_train))



    print("Tests complete")
