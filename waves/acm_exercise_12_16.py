# 12.16. Generate a “random” symmetric 2000 × 2000 matrix via the following scheme:
# pick a distribution (say, normal with a given mean and variance), and let the i, j entry
# with i ≥ j be an independent draw from this distribution. Let the remaining i < j entries
# be the symmetric mirror. Compute the eigenvalues of this matrix (which are all real) and
# plot them in a histogram. What does the result look like? How does this shape depend
# on the parameters of the distribution? On the choice of distribution?

import numpy as np
import matplotlib.pyplot as plt


distributions = {
    "normal": {
        "mean": 0,
        "std_dev": 1
    },
    "uniform": {
        "low": 0,
        "high": 1
    },
    "binomial": {
        "trials": 10,
        "probability_success": 0.5
    },
    "poisson": {
        "lambda_parameter": 10
    },
}


def generate_random_numbers_by_distribution(distribution, size):
    if distribution == "normal":
        samples = np.random.normal(distributions["normal"]
                                   ["mean"], distributions["normal"]["std_dev"], size)
        return samples
    elif distribution == "uniform":
        samples = np.random.uniform(
            low=distributions['uniform']['low'], high=distributions['uniform']['high'], size=size)
        return samples
    elif distribution == "binomial":
        samples = np.random.binomial(
            distributions['binomial']['trials'], distributions['binomial']['probability_success'], size)
        return samples
    elif distribution == "poisson":
        samples = np.random.poisson(distributions['poisson']['lambda_parameter'], size)
        return samples
    elif distribution == 'exponential':
        samples = np.random.exponential(distribution['exponential']['scale_parameter'], size)
        return samples


def exercise_12_16():
    n = 2000
      # Asume que tienes 4 distribuciones, ajusta según lo que necesites
    
    # Definir la cantidad de filas y columnas para los subplots
    rows = 2  # Ajusta según la cantidad de distribuciones
    cols = 2  # Ajusta según la cantidad de distribuciones

    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    for idx, dist in enumerate(distributions.keys()):
        matrix = generate_random_numbers_by_distribution(dist, n * n).reshape(n, n)
        for i in range(n):
            for j in range(i, n):
                matrix[j, i] = matrix[i, j]
        
        eigenvalues, _ = np.linalg.eig(matrix)
        ax = axes[idx//cols, idx%cols]
        ax.hist(eigenvalues, bins=50, edgecolor='black', alpha=0.7)
        ax.set_title(f'Histograma de Data, distribución {dist}')

    plt.tight_layout()
    plt.show()

exercise_12_16()
