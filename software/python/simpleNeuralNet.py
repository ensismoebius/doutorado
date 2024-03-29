import numpy as np


def nonlin(x, deriv=False):
    if deriv:
        return x*(1-x)
    return 1/(1+np.exp(-x))


# learnning rate
learnningRate = 0.1

# input data
inputs = np.array([
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
])

# output data
labels = np.array([
    [0],
    [1],
    [1],
    [0]
])

np.random.seed(1)

# Weights
syn0 = 2 * np.random.random((3, 4))-1
syn1 = 2 * np.random.random((4, 1))-1

# trainning step
for j in range(600000):
    l0 = inputs
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))

    l2_error = labels - l2
    l2_delta = l2_error*nonlin(l2, deriv=True)

    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error*nonlin(l1, deriv=True)

    if (j % 1000) == 0:
        print("Error " + str(np.mean(np.abs(l2_error))))

    # update synapses
    syn1 += l1.T.dot(l2_delta) * learnningRate
    syn0 += l0.T.dot(l1_delta) * learnningRate

print("Output after tranning")
print(l2)
