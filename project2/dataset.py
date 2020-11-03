import numpy as np

class Dataset:
    def __init__(self, image, regions):
        self.x = []
        self.y = []

        for c, region in enumerate(regions):
            region_img = image[region[0][0]:region[0][1], region[1][0]:region[1][1]]
            for y in range(region_img.shape[0]):
                for x in range(region_img.shape[1]):
                    colors = region_img[y, x]
                    features = self.get_features(colors)
                    self.x.append(features)
                    self.y.append(c)

        self.x = np.array(self.x)
        self.y = np.array(self.y)

    def get_features(self, colors):
        r = colors[0]
        g = colors[1]
        b = colors[2]
        sum = r+g+b
        return np.array([r/sum, g/sum, b/sum])

    def get_data(self):
        return self.x, self.y