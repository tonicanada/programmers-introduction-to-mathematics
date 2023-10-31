import os
import pickle
import numpy as np
import mnist_network as mn


def train_and_save_model_without_momentum(output_path):
    net = mn.train_mnist(os.path.join(os.path.dirname(__file__), 'mnist'), mn.build_network())
    with open(output_path, 'wb') as file:
        pickle.dump(net, file)


def evaluate_model(path_model):
    """
    Función que retorna el % de aciertos en el test_data set
    """
    with open(path_model, 'rb') as file:
        net = pickle.load(file)

    test_file = os.path.join(os.path.join(os.path.dirname(__file__), 'mnist', 'mnist_test.csv'))

    test_data = mn.load_1s_and_7s(test_file)

    n = len(test_data)
    evaluation_array = np.zeros(n)

    for i in range(n):
        output = net.evaluate((test_data[i][0]))
        if round(output) == test_data[i][1]:
            evaluation_array[i] = 1

    metric = np.mean(evaluation_array)

    print(metric)
    return metric


def create_network_with_momentum():
    input_nodes = mn.InputNode.make_input_nodes(28 * 28)

    first_layer = [mn.LinearNode(input_nodes) for i in range(10)]
    first_layer_relu = [mn.ReluNode(L) for L in first_layer]

    second_layer = [mn.LinearNode(first_layer_relu) for i in range(10)]
    second_layer_relu = [mn.ReluNode(L) for L in second_layer]

    linear_output = mn.LinearNode(second_layer_relu)
    output = mn.SigmoidNode(linear_output)
    error_node = mn.L2ErrorNode(output)
    network = mn.NeuralNetwork(
        output, input_nodes, error_node=error_node, step_size=0.05, momentum_decay=0.5)

    return network


def train_and_save_model_with_momentum(output_path):
    net = mn.train_mnist(os.path.join(os.path.dirname(__file__), 'mnist'),
                         create_network_with_momentum())
    with open(output_path, 'wb') as file:
        pickle.dump(net, file)


def load_model(input_path):
    with open(input_path, 'rb') as file:
        net = pickle.load(file)
    return net


# train_and_save_model_without_momentum(
#     "./neural_network/acm_models/exercise_14_14/20231002_minst_without_momentum.pkl")


# train_and_save_model_with_momentum(
#     "./neural_network/acm_models/exercise_14_14/20231002_minst_with_momentum.pkl")


# evaluate_model("./neural_network/acm_models/exercise_14_14/20231002_minst_without_momentum.pkl")
# evaluate_model("./neural_network/acm_models/exercise_14_14/20231002_minst_with_momentum.pkl")
net = load_model("./neural_network/acm_models/exercise_14_14/20231002_minst_with_momentum.pkl")
test_file = os.path.join(os.path.join(os.path.dirname(__file__), 'mnist', 'mnist_test.csv'))
