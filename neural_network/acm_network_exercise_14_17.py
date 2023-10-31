import os
import pickle
import numpy as np
import mnist_network as mn
import acm_neural_network_softmax as nn_acm


def build_network():
    input_nodes = nn_acm.InputNode.make_input_nodes(28 * 28)

    first_layer = [nn_acm.LinearNode(input_nodes) for i in range(10)]
    first_layer_relu = [nn_acm.ReluNode(L) for L in first_layer]

    second_layer = [nn_acm.LinearNode(first_layer_relu) for i in range(10)]
    second_layer_relu = [nn_acm.ReluNode(L) for L in second_layer]
    
    output = nn_acm.SoftMax(second_layer_relu)

    error_node = nn_acm.CrossEntropyErrorNode(output)

    network = nn_acm.NeuralNetwork(
        output, input_nodes, error_node=error_node, step_size=0.05)

    return network


def train_mnist(data_dirname, network, num_epochs=5):
    train_file = os.path.join(data_dirname, 'mnist_train.csv')
    test_file = os.path.join(data_dirname, 'mnist_test.csv')
    try:
        train = mn.load_all_data(train_file, size=10000)
        test = mn.load_all_data(test_file)
    except Exception:  # pragma: no cover
        print(mn.cant_find_files.format(train_file, test_file))
        raise

    n = len(train)
    epoch_size = int(n / 10)

    for i in range(num_epochs):
        mn.shuffle(train)
        validation = train[:epoch_size]
        real_train = train[epoch_size: 2 * epoch_size]

        print("Starting epoch of {} examples with {} validation".format(
            len(real_train), len(validation)))

        network.train(real_train, max_steps=len(real_train))

        print("Finished epoch. Validation error={:.3f}".format(
            network.error_on_dataset(validation)))

    print("Test error={:.3f}".format(network.error_on_dataset(test)))
    mn.show_random_examples(network, test)
    return network


net = build_network()
train_mnist(os.path.join(os.path.dirname(__file__), 'mnist'), net)
