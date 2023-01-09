"""
Assignment Title: Knearest Neighbor (KNN) and Perceptron methods
Purpose         : Implementation of both methods
Language        : Python
Author          : Hemant Ramphul
Github          : https://github.com/hemantramphul/Knearest-Neighbor-KNN-and-Perceptron-methods
Date            : 03 January 2023

Université des Mascareignes (UdM)
Faculty of Information and Communication Technology
Master Artificial Intelligence and Robotics
Official Website: https://udm.ac.mu
"""

import numpy as np
import matplotlib.pyplot as plt


def active(x, w):
    """
    Function to return sum between x(i) w(i+1).

    :param x: prediction.
    :param w: weights in an array.
    :return: final result of res.
    """
    res = 0
    for i in range(0, 2):
        # Calculate the sum of the products between the rows x(i) with the rows w(i+1)(i+1) for w because the first row contains the threshold)
        res += x[0, i] * w[0, i + 1]

    res += w[
        0, 0]  # After the calculation of the products between x(i) w(i+1), we add the threshold with the result res
    return res  # Return the final result of res


def perceptron(x, w, active):
    """
    Function to do the prediction.

    :param x: prediction.
    :param w: weights in an array.
    :param active: indicates the activation function used.
    :return: Prediction -1 or 1
    """
    y = active(x, w)  # Gets the result of the active function (output of the neuron)
    if y < 0:
        return -1  # If the active gives a negative number then returns -1 (the prediction)
    else:
        return 1  # Else return 1


def apprentissage(data, oracle, active):
    """
    Function to train.

    :param data: dataset to train.
    :param oracle: oracle.
    :param active: indicates the activation function used.
    :return: full pass of training set.
    """
    # Initialization of parameters with 2 weights in an array (neuron)
    w = np.array([[0.5, 1.2, 0.8]])
    # Initialization alpha learning rate
    alpha = 0.1
    # Initialize an empty (x) array that takes both inputs to do the calculations if the prediction is wrong
    x = np.array([[0, 0]])

    mdiff = []  # Initialize an error table for calculating the cumulative error

    for j in range(0, 100):
        mdiff += [0]  # Add a flag to the error array with each iteration
        for i in range(len(data[0])):
            x[0, 0] = data[0][i]  # Store input in x
            x[0, 1] = data[1][i]
            h = perceptron(x, w, active)  # Call of the perceptron function to make the prediction
            mdiff[j] += (oracle[i] - h) ** 2  # Calculate the error

            if h != oracle[i]:  # If the prediction is made, we update the parameters
                w[0, 1] += alpha * (oracle[i] - h) * x[0, 0]  # update w1
                w[0, 2] += alpha * (oracle[i] - h) * x[0, 1]  # update w2
                w[0, 0] += alpha * (oracle[i] - h)  # update of threshold

    # Return parameters after update with cumulative error for full pass of training set
    return w, mdiff


def affiche_classe(x, clas, K, w, mdiff):
    """
    Function that allows graphical visualization in a planar representation of the result of the classification.

    :param x: indicates test data.
    :param clas: indicates the class number of the individual.
    :param K: indicates the number of neighbors used in the algorithm.
    :param w: weights in an array.
    :param mdiff: cumulative error.
    :return: show plot graphical.
    """
    plt.figure(figsize=(15, 6), dpi=80)  # Figure size
    plt.subplot(1, 2, 1)  # Row 1, Col 1 , Index 1
    plt.title('Errors Plot')  # Plot title
    plt.plot(mdiff)

    t = [np.min(x[0, :]), np.max(x[0, :])]
    z = [(-w[0, 0] - w[0, 1] * np.min(x[0, :])) / w[0, 2], (-w[0, 0] - w[0, 1] * np.max(x[0, :])) / w[0, 2]]

    plt.subplot(1, 2, 2)  # Row 1, Col 2 , Index 2
    plt.title("ML Plot")  # Plot title
    plt.plot(t, z)
    ind = (clas == -1)
    plt.plot(x[0, ind], x[1, ind], "o")
    ind = (clas == 1)
    plt.plot(x[0, ind], x[1, ind], "o")
    plt.show()  # Show the final result


# Données de test
mean1 = [4, 4]
cov1 = [[1, 0], [0, 1]]
data1 = np.transpose(np.random.multivariate_normal(mean1, cov1, 128))

mean2 = [-4, -4]
cov2 = [[4, 0], [0, 4]]
data2 = np.transpose(np.random.multivariate_normal(mean2, cov2, 128))

data = np.concatenate((data1, data2), axis=1)
oracle = np.concatenate((np.zeros(128) - 1, np.ones(128)))

w, mdiff = apprentissage(data, oracle, active)

affiche_classe(data, oracle, 2, w, mdiff)
