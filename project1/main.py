import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class NearestNeighbour:
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def predict(self, x):
        # Pythonic og effektivt, men...spaghetti?
        return min(enumerate(self.y_data), key=lambda xi: np.sum((x-self.x_data[xi[0]])**2))[1]

if __name__ == "__main__":
    data = pd.read_csv('data/ds-1.txt', sep="  ", header=None)
    data.columns = ["class", "feature_1", "feature_2", "feature_3", "feature_4"]
    print(data.head(10))
    x = data.iloc[:, 1:].to_numpy()
    y = data.iloc[:, 0].to_numpy()
    plt.scatter(x[:,0], x[:,1], c = y)
    plt.show()
    plt.scatter(x[:,1], x[:,2], c = y)
    plt.show()
    model = NearestNeighbour(x,y)
    """
        Tests
    """
    for i in range(len(x)):
        assert(model.predict(x[i])==y[i], "Nearest Neigbour Classifier not predicting correctly")
