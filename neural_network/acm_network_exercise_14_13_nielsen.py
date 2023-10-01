# 14.13. Write a program that uses gradient descent to learn linear threshold functions. In
# particular: write a function that samples data uniformly from the set [0, 1]5 ⊂ R5 , and
# labels them (unbeknownst to the learning algorithm) according to their value under a
# fixed linear threshold function Lw,b . Design a learning algorithm to learn w and b from
# the data. That is, determine what the appropriate loss function should be, determine a
# formula for the gradient, and enshrine it in code. How much data is needed to successfully
# and consistently learn? How does this change as the exponent 5 grows?

import numpy as np
import random

import acm_network_from_nielsenbook as nielsen


def threshold_function(input_array, threshold, w=np.array([1, -2, 2, 3, 0.5]), b=0.5):
    output = np.dot(input_array, w) + b
    if output >= threshold:
        return 1
    else:
        return 0


def generate_data(set_size=20000, seed=42):
    random.seed(seed)
    data_input = np.zeros((set_size, 5))
    data_ouput = np.zeros(set_size)

    for i in range(set_size):
        input = [random.uniform(0, 1) for _ in range(5)]
        data_input[i] = input
        data_ouput[i] = threshold_function(input, 2.5)

    data_input = [np.reshape(x, (5, 1)) for x in data_input]
    test_data = list(zip(data_input, data_ouput))
    return test_data


def train_and_save_model(path, sizes=[5, 5, 2]):
    training_data = generate_data(seed=42)
    test_data = generate_data(seed=30)
    net = nielsen.Network(sizes)
    net.SGD(training_data, 100, 10, 0.18, test_data=test_data)
    net.save_model(path)


def evaluate_model(path_model, test_data):
    """
    Función que retorna el % de aciertos en el test_data set
    """
    net = nielsen.Network.load_model(path_model)
    n = len(test_data)
    evaluation_array = np.zeros(n)

    for i in range(n):
        output = np.argmax(net.feedforward(test_data[i][0]))
        if output == test_data[i][1]:
            evaluation_array[i] = 1

    metric = np.mean(evaluation_array)

    print(metric)
    return metric


# train_and_save_model("./neural_network/acm_models/20230928_nielsenmodel_4.pkl")


test_data = generate_data(seed=50)
evaluate_model("./neural_network/acm_models/20230928_nielsenmodel_3.pkl", test_data)
