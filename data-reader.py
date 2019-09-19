import numpy as np
import matplotlib.pyplot as plt

N = 6040  # number of users
M = 3952  # number of movies


def read_data():
    data = np.zeros((N, M))
    f = open("./data/ratings.dat", "r")
    lines = f.readlines()

    for line in lines:
        user, movie, rating, *_ = list(map(int, line.split("::")))
        data[user-1, movie-1] = rating

    return data


def get_indicators(N, M, prob_std=0.5):
    ind = np.random.binomial(1, prob_std, (N, M))
    return ind


data = read_data()
I = get_indicators(N, M)
print(data)
print(I)
train_data = np.where(I == 1, data, np.zeros((N, M)))
test_data = np.where(I == 0, data, np.zeros((N, M)))
print(train_data.shape)
print(train_data)
plt.imshow(data, vmin=np.min(data), vmax=np.max(data))
plt.show()
plt.imshow(train_data, vmin=np.min(train_data), vmax=np.max(train_data))
plt.show()
plt.imshow(test_data, vmin=np.min(test_data), vmax=np.max(test_data))
plt.show()