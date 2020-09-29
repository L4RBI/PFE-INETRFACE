import numpy
from scipy.spatial.distance import cdist
from tools import *

n = 0

# Partition Coefficient


def pc(membership):
    N = membership.shape[n] + 1
    M = numpy.power(membership, 2)
    M = numpy.sum(M)
    return M/N

# Classification Entropy


def ce(membership, m=2):
    N = membership.shape[n] + 1
    M = membership
    LM = numpy.log(M)
    return - numpy.sum(numpy.multiply(M, LM)) / N

#xie and beni


def xb(histogram, centers, m=2):
    _membership = membership(histogram=histogram, centers=centers, m=m)
    N = _membership.shape[n] + 1
    M = numpy.power(_membership, m)
    Dxv = cdist(histogram, centers)
    Dvv = cdist(centers, centers)
    Dvv[Dvv == 0] = numpy.inf
    return numpy.sum(numpy.multiply(M, Dxv)) / (N * numpy.min(Dvv))

#


def sc(histogram, centers, m):
    _membership = membership(histogram=histogram, centers=centers, m=m)
    Ni = numpy.bincount(numpy.argmax(_membership, axis=-1))
    M = numpy.power(_membership, 2)
    Dxv = cdist(histogram, centers)
    DVV = cdist(centers, centers)
    MDxv = numpy.multiply(M, Dxv)
    temp = []
    for i in range(len(Ni)):
        temp.append(numpy.sum(MDxv[i])/(Ni[i]*numpy.sum(DVV[i])))
    return numpy.sum(temp)
