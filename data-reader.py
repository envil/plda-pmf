import numpy as np

data = np.zeros((6040 + 1, 3952 + 1))
f = open("./data/ratings.dat", "r")
lines = f.readlines()

for line in lines:
    user, movie, rating, *_ = list(map(int, line.split("::")))
    data[user, movie] = rating
