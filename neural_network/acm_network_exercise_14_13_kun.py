import mnist_network as mn
import random
import numpy as np
import pickle


def threshold_function(input_array, threshold, w=np.array([1, -2, 2, 3, 0.5]), b=0.5):
    output = np.dot(input_array, w) + b
    if output >= threshold:
        return 1
    else:
        return 0


def generate_data(set_size=20000, seed=42):
    random.seed(seed)
    test_data = []

    for i in range(set_size):
        input = [random.uniform(0, 1) for _ in range(5)]
        ouput = threshold_function(input, 2.5)
        test_data.append([input, ouput])

    return test_data


def build_network():
    input_nodes = mn.InputNode.make_input_nodes(5)

    first_layer = [mn.LinearNode(input_nodes) for i in range(5)]
    first_layer_relu = [mn.ReluNode(L) for L in first_layer]

    second_layer = [mn.LinearNode(first_layer_relu) for i in range(10)]
    second_layer_relu = [mn.ReluNode(L) for L in second_layer]

    linear_output = mn.LinearNode(second_layer_relu)
    output = mn.SigmoidNode(linear_output)
    error_node = mn.L2ErrorNode(output)

    network = mn.NeuralNetwork(
        output, input_nodes, error_node=error_node, step_size=0.05)

    return network


def train_mnist(training_data, test_data, num_epochs=5):

    network = build_network()
    training_data = generate_data(seed=42)
    test_data = generate_data(seed=32)

    n = len(training_data)
    epoch_size = int(n / 10)

    for i in range(num_epochs):
        random.shuffle(training_data)
        validation = training_data[:epoch_size]
        real_train = training_data[epoch_size: 2 * epoch_size]

        print("Starting epoch of {} examples with {} validation".format(
            len(real_train), len(validation)))

        network.train(real_train, max_steps=len(real_train))

        print("Finished epoch. Validation error={:.3f}".format(
            network.error_on_dataset(validation)))

    print("Test error={:.3f}".format(network.error_on_dataset(test_data)))
    mn.show_random_examples(network, test_data)
    return network


def train_and_save_model(path):
    training_data = generate_data(seed=42)
    test_data = generate_data(seed=32)
    net = train_mnist(training_data, test_data)
    with open(path, 'wb') as file:
        pickle.dump(net, file)


def evaluate_model(path_model, test_data):
    """
    Función que retorna el % de aciertos en el test_data set
    """
    with open(path_model, 'rb') as file:
        net = pickle.load(file)

    n = len(test_data)
    evaluation_array = np.zeros(n)

    for i in range(n):
        output = net.evaluate((test_data[i][0]))
        if round(output) == test_data[i][1]:
            evaluation_array[i] = 1

    metric = np.mean(evaluation_array)

    print(metric)
    return metric


# build_network()
# train_mnist()

train_and_save_model("./neural_network/acm_models/20231006_kunmodel_1.pkl")


# test_data = generate_data(seed=50)
# evaluate_model("./neural_network/acm_models/exercise_14_13/20230930_kunmodel_1.pkl", test_data)
