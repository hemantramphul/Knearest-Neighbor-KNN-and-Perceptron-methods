"""
Assignment Title: Méthodes des Kplus proche Voisins et du Perceptron
Purpose         : Implementation of both methods
Language        : Python
Author          : Hemant Ramphul
Github          : https://github.com/hemantramphul/Knearest-Neighbor-KNN-and-Perceptron-methods
Date            : 02 January 2023

Université des Mascareignes (UdM)
Faculty of Information and Communication Technology
Master Artificial Intelligence and Robotics
Official Website: https://udm.ac.mu
"""

import numpy as np
import matplotlib.pyplot as plt


def kppv(x, appren, oracle, K):
    """
    Function to return the predicted class.

    :param x: number of data per class in dataset.
    :param appren: number of classes to recognize.
    :param oracle: dimensionality of dataset.
    :param K: prediction with parameter K.
    :return: predicted class.
    """
    # Initialization of a class matrix containing 0 to store the prediction of the class of each point
    clas = np.zeros(len(x[0]))

    for i in range(0, len(x[0])):
        # Initialization of a result table to memorize the Euclidean distances of a point to be predicted with all
        # the points of the learning data
        result = []

        for j in range(0, len(appren[0])):
            # Calculate euclidean distance between two points
            distance = euclidean_distance(x[0][i], appren[0][j], x[1][i], appren[1][j])
            # Store the result of the previous distance calculation with the class of the point of the learning data
            # in the result array
            result.append((distance, oracle[j]))
        result.sort()  # Sort the result array to take K the smallest distances

        cl0 = 0  # Counter for class 0
        cl1 = 0  # Counter for class 1

        # Iterate over the K points in the result array to calculate the number of each class between these K points
        for c in result[0:K]:
            if c[1] == 1:
                cl1 += 1
            else:
                cl0 += 1

        if cl1 > cl0:  # If the number of class1 is greater than class0 then we predict class 1
            clas[i] = 1
        else:  # Else predict class0
            clas[i] = 0

    return clas


def euclidean_distance(x1, x2, y1, y2):
    """
    Function to calculate the Euclidean distance between two points.

    Euclidean Distance Formula: distance = √[(x 2 – x 1 )2 + (y 2 – y 1 )2]

    :param x1: is the coordinate of the first point.
    :param x2: is the coordinate of the second point.
    :param y1: is the coordinate of the first point.
    :param y2: is the coordinate of the second point.
    :return: distance
    """
    return np.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))


def affiche_classe(x, clas, K):
    """
    Function that allows graphical visualization in a planar representation of the result of the classification.

    :param x: indicates test data.
    :param clas: indicates the class number of the individual.
    :param K: indicates the number of neighbors used in the algorithm.
    :return: figure
    """
    for k in range(0, K):
        ind = (clas == k)  # Check if class duplicates
        plt.plot(x[0, ind], x[1, ind], "o")  # Plot coordinates
    plt.show()  # Show the figure in a window


# Donnees de test
mean1 = [4, 4]
cov1 = [[1, 0], [0, 1]]
data1 = np.transpose(np.random.multivariate_normal(mean1, cov1, 128))

mean2 = [-4, -4]
cov2 = [[4, 0], [0, 4]]
data2 = np.transpose(np.random.multivariate_normal(mean2, cov2, 128))

data = np.concatenate((data1, data2), axis=1)
oracle = np.concatenate((np.zeros(128), np.ones(128)))

test1 = np.transpose(np.random.multivariate_normal(mean1, cov1, 64))
test2 = np.transpose(np.random.multivariate_normal(mean2, cov2, 64))

test = np.concatenate((test1, test2), axis=1)

K = 3
clas = kppv(test, data, oracle, K)
affiche_classe(test, clas, 3)
