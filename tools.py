from matplotlib.image import imread
import matplotlib.pyplot as plt
from math import sqrt
import math
import random
import numpy
import operator
from scipy.spatial.distance import cdist
from scipy.linalg import norm

# computes the histogram of an image


def Histogram(path):
    image = imread(path)
    if len(image.shape) != 2:
        def gray(rgb): return numpy.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])
        gray = gray(image)
        image = gray
    hist, bins = numpy.histogram(image.ravel(), 256, [0, 256])
    return adapt(hist)  # changed

# plots a histogram


def HistogramV2(path):
    image = imread(path)
    if len(image.shape) != 2:
        def gray(rgb): return numpy.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])
        gray = gray(image)
        image = gray
    hist, bins = numpy.histogram(image.ravel(), 256, [0, 256])
    return hist


def ShowHistogram(histogram):
    plt.plot(range(0, 256), histogram)
    plt.xlabel('Grey intensity lvl')
    plt.ylabel('Frequency')
    plt.title('Histogram')
    plt.show()

# adapts a histogram


def adapt(histogram):
    adapted_histogram = numpy.zeros((256, 2))
    for i in range(256):
        adapted_histogram[i][0] = i
        adapted_histogram[i][1] = histogram[i]
    return adapted_histogram

# calculates the euclidean distance between two points


def Dis(x, y):
    dist = sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)
    return dist

# intra distance not yet fixed


def instraDistance(data, membership, centers):
    membership = numpy.argmax(membership, axis=-1)
    temp = []
    for i in range(len(membership)):
        print(data[i])
        #x = Dis(data[i],centers[membership[i]])
        # temp.append(x)
    temp = numpy.power(temp, 2)
    return numpy.sum(temp)

# calculates the Membership matrix


def membership(histogram, centers, m):
    U_temp = cdist(histogram, centers, 'euclidean')
    U_temp = numpy.power(U_temp, 2/(m - 1))
    denominator_ = U_temp.reshape(
        (histogram.shape[0], 1, -1)).repeat(U_temp.shape[-1], axis=1)
    denominator_ = U_temp[:, :, numpy.newaxis] / denominator_
    return 1 / denominator_.sum(2)

# The fcm objective function


def J(histogram, membership, centers, m):
    distance = cdist(histogram, centers, 'euclidean')
    distance = numpy.power(distance, 2)
    membership = numpy.power(membership, 2)
    return numpy.sum(numpy.multiply(distance, membership))
