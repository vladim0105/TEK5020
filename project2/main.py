import numpy as np
import matplotlib.pyplot as plt
from project2.dataset import Dataset
if __name__ == "__main__":
    img1 = plt.imread("data/Bilde1.png")
    img1_regions = [
        [[70, 170], [200, 300]], #Red paprika
        [[350,450], [250,350]], # Dark green paprika
        [[150, 400], [100, 160]] # Light green paprika/chili whatever
    ]
    img2 = plt.imread("data/Bilde2.png")
    img2_regions = [
        [[350, 550], [225, 500]], #Blue folder, with metal
        [[200,470], [750,1000]], # Red folder, with metal and papers
        [[100, 300], [100, 300]] # Floor
    ]
    img3 = plt.imread("data/Bilde3.png")
    img3_regions = [
        [[50, img3.shape[1]-80], [100, img3.shape[1]-100]] # A region which excludes the thick, white, outline
    ]

    img1_train_dataset= Dataset(img1, img1_regions)
    train_x, train_y = img1_train_dataset.get_data()

    #Train om img1 and test on img1 here

    img2_train_dataset= Dataset(img2, img2_regions)
    train_x, train_y = img2_train_dataset.get_data()

    # Train om img2 here

    img2_test_dataset= Dataset(img3, img3_regions)
    test_x, test_y = img2_test_dataset.get_data()

    # Test on img3 here