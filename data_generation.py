import numpy as np
import pandas as pd
from scipy.stats import gamma


def create_dataset(m, t, theta):
    """
    Creates a dataset based on the provided parameters.

    Parameters:
    m (int): Number of components in the dataset.
    t (list): List of time points.
    theta (list): List of parameters including alpha, eta, beta, mu_A, sigma_A, sigma_Z.

    Returns:
    numpy.ndarray: An array representing the generated dataset.
    """
    alpha, eta, beta, mu_A, sigma_A, sigma_Z = theta
    Y = []
    A = []

    for comp in range(m):
        Y_j = []
        A_j = rng.normal(mu_A, sigma_A)
        A.append(A_j)
        X_prev = 0

        for i in range(len(t)):
            if i == 0:
                alpha_t1 = 0;
                alpha_t2 = t[0]
            else:
                alpha_t1 = t[i - 1]
                alpha_t2 = t[i]

            alpha_tmp = alpha * (alpha_t2 ** eta) - alpha * (alpha_t1 ** eta)
            X_t = rng.gamma(alpha_tmp, beta) + X_prev
            X_prev = X_t
            Z_t = rng.normal(0, sigma_Z)
            Y_t = A_j + X_t + Z_t
            Y_j.append(Y_t)

        Y.append(Y_j)

    Y = np.array(Y)
    return Y


def create_datasets(true_thetas, data_seed):
    """
    Generates seeded stationary and non-stationary GP datasets based on true parameters.

    Parameters:
    true_thetas (dict): A dictionary containing the true thetas for different cases.
    data_seed (int): The seed for data generation.

    Returns:
    None
    """
    ### Stationary GP data (Herr2023/Hazra2020 case study 1)
    m = 5
    t = [5, 10, 15]
    df = pd.DataFrame(create_dataset(m, t, theta=true_thetas["A"]))
    df.columns = t
    df.to_csv("./data/case_A.csv", index=False)

    ### Non-stationary GP data (Herr2023/Hazra2020 case study 2)
    m = 10
    t = [2, 4, 6]
    df = pd.DataFrame(create_dataset(m, t, theta=true_thetas["B"]))
    df.columns = t
    df.to_csv("./data/case_B.csv", index=False)

    with open('./data/data_seed.txt', 'w') as f:
        f.write(f"{data_seed}")


if __name__ == '__main__':
    data_seed = 4
    rng = np.random.default_rng(data_seed)
    true_thetas = dict({"A": [4, 1, 0.015, 0.0, 0.0, 0.1],
                        "B": [2, 2.5, 0.01, 0.5, 0.1, 0.1],
                        "C": [4.4, 1.07, 0.04, -4.5, 0.2, 1.3]})  # Approx.

    create_datasets(true_thetas, data_seed)
