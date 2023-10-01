import mnist_network as mn
import random
import numpy as np


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

    linear_output = mn.LinearNode(first_layer_relu)
    output = mn.SigmoidNode(linear_output)
    error_node = mn.L2ErrorNode(output)

    print(error_node.arguments[0].evaluate([1,2,6,4,5]))

    network = mn.NeuralNetwork(
        output, input_nodes, error_node=error_node, step_size=0.05)

    return network


def train_mnist(num_epochs=5):

    network = build_network()
    training_data = generate_data(seed=42)
    n = len(training_data)
    epoch_size = int(n / 10)

    for i in range(num_epochs):
        random.shuffle(training_data)
        validation = training_data[:epoch_size]
        real_train = training_data[epoch_size: 2 * epoch_size]

        print("Starting epoch of {} examples with {} validation".format(
            len(real_train), len(validation)))

        network.train(real_train, max_steps=len(real_train))

    #     print("Finished epoch. Validation error={:.3f}".format(
    #         network.error_on_dataset(validation)))

    # print("Test error={:.3f}".format(network.error_on_dataset(test)))
    # show_random_examples(network, test)
    return network


build_network()
# train_mnist()

