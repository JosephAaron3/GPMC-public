import os
import time

import numpy as np
import pandas as pd
from scipy.stats import gamma
from tqdm import tqdm


def load_data(case, fp):
    """
    Reads data from a CSV file, processes it based on the case type, and returns the processed data.

    Parameters:
    - case: A string indicating the case ("A", "B", or "C").
    - fp: A string representing the file path to the CSV file.

    Returns:
    A tuple containing:
    - t: A list of integers representing time points.
    - n: An integer representing the number of time points.
    - m: An integer representing the number of components.
    - Y: A NumPy array containing the processed data.
    """
    df = pd.read_csv(fp)
    Y = df.values
    t = [0]
    n = 0
    for col in df.columns:
        t.append(int(col))
        n = n + 1
    m = Y.shape[0]
    if case == "C":  # Edit due to data including time 0
        Y = 100 - Y[:, 1:]
        t = t[1:]
        n -= 1
    Y = np.hstack([np.zeros(m).reshape(-1, 1), Y])
    return [t, n, m], Y


def get_parameters(case_studies, theta_set):
    """
    A function to get parameters for the given case studies and theta set.

    Parameters:
    - case_studies: A list of case study identifiers.
    - theta_set: A string specifying the set of thetas to use.

    Returns:
    - data_seed: An integer representing the data seed.
    - datasets: A list of dictionaries containing the name, parameters, and data for each case study.
    - thetas: A dictionary containing the theta values for each case study.
    """
    # Data seed
    with open("./data/data_seed.txt", "r") as f:
        data_seed = int(f.readline())

    # Datasets
    datasets = []
    for case in case_studies:
        params, data = load_data(case, f"./data/case_{case}.csv")
        datasets.append(dict({"name": case, "params": params, "data": data}))

    # Thetas
    true_thetas = dict({"A": [4, 1, 0.015, 0.0, 0.0, 0.1],
                        "B": [2, 2.5, 0.01, 0.5, 0.1, 0.1],
                        "C": [4.4, 1.07, 0.04, -4.5, 0.2, 1.3]})  # Approx.

    thetas = dict({"A": [true_thetas["A"].copy()],
                   "B": [true_thetas["B"].copy()],
                   "C": [true_thetas["C"].copy()]})

    if theta_set == "All":
        theta_ranges = dict({"A": [np.linspace(3.000, 5.000, 5),
                                   np.linspace(0.800, 1.200, 5),
                                   np.linspace(0.010, 0.020, 5),
                                   np.linspace(-0.20, 0.200, 5),
                                   np.linspace(0.000, 0.100, 5),
                                   np.linspace(0.050, 0.150, 5)],
                             "B": [np.linspace(1.000, 3.000, 5),
                                   np.linspace(2.300, 2.700, 5),
                                   np.linspace(0.005, 0.015, 5),
                                   np.linspace(0.300, 0.700, 5),
                                   np.linspace(0.050, 0.150, 5),
                                   np.linspace(0.050, 0.150, 5)],
                             "C": [np.linspace(2.400, 6.400, 5),
                                   np.linspace(0.900, 1.240, 5),
                                   np.linspace(0.015, 0.065, 5),
                                   np.linspace(-5.50, -3.50, 5),
                                   np.linspace(0.100, 0.300, 5),
                                   np.linspace(1.000, 1.600, 5)]})
        for case in theta_ranges.keys():
            for i, param_range in enumerate(theta_ranges[case]):
                for param in param_range:
                    theta_temp = true_thetas[case].copy()
                    theta_temp[i] = round(param, 5)
                    if theta_temp not in thetas[case]:
                        thetas[case].append(theta_temp)
    elif theta_set != "True":
        print("Chosen theta_set not an option")
        exit()

    return data_seed, datasets, thetas


def get_sigmas(sig, n):
    """
    Generates the determinant and inverse of the sigma matrix.

    Args:
        sig (float): The sigma value.
        n (int): The size of the matrix.

    Returns:
        tuple: The determinant and inverse of the Sigma matrix.
    """
    A = np.eye(n)
    A[0, 1] = -0.5  # First row
    A[n - 1, n - 2] = -0.5  # Last row
    for i in range(1, n - 1):  # Other rows
        A[i, i - 1] = -0.5
        A[i, i + 1] = -0.5

    Sigma_Z = 2 * sig ** 2 * A
    DeterminantSigma_Z = np.linalg.det(Sigma_Z)
    InvSigma_Z = np.linalg.inv(Sigma_Z)

    return DeterminantSigma_Z, InvSigma_Z


class CMC2Algorithm:
    def __init__(self, Y, t, n, m, N, theta, run_seed):
        """
        Initializes the CMC2Algorithm with the provided parameters.

        Args:
            Y: Data matrix.
            t: Time points.
            n: Number of time points.
            m: Number of components.
            N: Number of Monte Carlo samples.
            theta: Model parameters.
            run_seed: Random seed for the RNG.

        Returns:
            None
        """
        rng = np.random.default_rng(run_seed)
        self.Y = Y
        self.n = n
        self.m = m
        self.N = N
        self.theta = theta
        self.DeterminantSigma_Z, self.InvSigma_Z = get_sigmas(theta[5], n)
        self.alphas = np.array([theta[0] * t[j] ** theta[1] - theta[0] * t[j - 1] ** theta[1]
                                for j in range(1, n + 1)])
        self.A = rng.normal(theta[3], theta[4], size=(m, N))
        self.deltaX = np.array([[rng.gamma(self.alphas[j], theta[2], size=N) for j in range(n)]
                                for _ in range(m)])

    def run(self, algorithm):
        """
        Runs a specified algorithm and calculates various output statistics.

        Args:
            algorithm: A string specifying the algorithm to run.

        Returns:
            A list containing ell_hat (estimate), var (MC variance), re (MC relative error),
            runtime (in seconds), and rtvp (MC relative time-variance product).
        """
        start_time = time.time()
        if algorithm == "Original":
            ell_hat, var, re = self._run_original()
        elif algorithm == "Product":
            ell_hat, var, re = self._run_product()
        elif algorithm == "Original_CVs":
            ell_hat, var, re = self._run_original_cvs()
        elif algorithm == "Product_CVs":
            ell_hat, var, re = self._run_product_cvs()
        runtime = time.time() - start_time
        rtvp = re ** 2 * runtime
        return [ell_hat, var, re, runtime, rtvp]

    def _mvn_pdf_log(self, x, mu):
        """
        Calculates the log of the multivariate normal probability density function at a given
        input vector x and mean vector mu.

        Args:
            x: Input vector of values.
            mu: Mean vector.

        Returns:
            Log of the multivariate normal probability density function.
        """
        return - len(x) / 2 * np.log(2 * np.pi) - 0.5 * np.log(self.DeterminantSigma_Z) - \
            0.5 * np.matmul(np.matmul((x - mu), self.InvSigma_Z), np.transpose(x - mu))

    def _cv_fn(self, x, y):
        """
        Calculates the control variables (i.e. the sum of absolute differences)

        Args:
            x: An array representing the first input.
            y: An array representing the second input.

        Returns:
            The sum of absolute differences between the subarrays x[1:] and y[1:].
        """
        return sum(abs(x[1:] - y[1:]))

    def _cv_mean_fn(self, y):
        """
        Calculates the mean of the control variables for a given input y.

        Args:
            y: An array representing the input values.

        Returns:
            The mean of the control variables calculated based on the input y.
        """
        return sum(self.alphas[i] * self.theta[2] - y[i] - 2 * self.alphas[i] * self.theta[2] * \
                   gamma.cdf(y[i], a=self.alphas[i] + 1, scale=self.theta[2]) + \
                   2 * y[i] * gamma.cdf(y[i], a=self.alphas[i], scale=self.theta[2])
                   for i in range(1, self.n))

    def _run_original(self):
        """
        Runs the original algorithm and calculates various output statistics.

        Returns:
            Ell_hat (float): MC estimate.
            Ell_var (float): MC variance.
            Ell_re (float): MC relative error.
        """
        ell = np.zeros(self.N)
        for k in range(self.N):
            for i in range(self.m):
                self.Y[i, 0] = self.A[i, k]
                deltaY = np.diff(self.Y[i, :])
                ell[k] += self._mvn_pdf_log(deltaY, self.deltaX[i, :, k])
        Ell_hat = - np.log(self.N) + np.max(ell) + np.log(np.sum(np.exp(ell - np.max(ell))))
        Ell_var = np.var(np.exp(ell)) / self.N
        Ell_re = np.sqrt(Ell_var) / np.exp(Ell_hat)
        return Ell_hat, Ell_var, Ell_re

    def _run_product(self):
        """
        Runs the product algorithm and calculates various output statistics.

        Returns:
            Ell_hat (float): MC estimate.
            Ell_var (float): MC variance.
            Ell_re (float): MC relative error.
        """
        ell_hat = np.zeros(self.m)
        ell_hat_var = np.zeros(self.m)
        for i in range(self.m):
            ell = np.zeros(self.N)
            for k in range(self.N):
                self.Y[i, 0] = self.A[i, k]
                deltaY = np.diff(self.Y[i, :])
                ell[k] = self._mvn_pdf_log(deltaY, self.deltaX[i, :, k])
            ell_hat[i] = - np.log(self.N) + np.max(ell) + np.log(np.sum(np.exp(ell - np.max(ell))))
            ell_hat_var[i] = np.var(np.exp(ell)) / self.N
        Ell_hat = np.sum(ell_hat)
        Ell_var = np.prod(ell_hat_var + np.exp(ell_hat) ** 2) - np.prod(np.exp(ell_hat) ** 2)
        Ell_re = np.sqrt(Ell_var) / np.exp(Ell_hat)
        return Ell_hat, Ell_var, Ell_re

    def _run_original_cvs(self):
        """
        Runs the original algorithm with control variables and calculates various output statistics.

        Returns:
            Ell_hat (float): MC estimate.
            Ell_var (float): MC variance.
            Ell_re (float): MC relative error.
        """
        ell = np.zeros(self.N)
        ell_cv = np.zeros(self.N)
        for k in range(self.N):
            for i in range(self.m):
                self.Y[i, 0] = self.A[i, k]
                deltaY = np.diff(self.Y[i, :])
                ell[k] += self._mvn_pdf_log(deltaY, self.deltaX[i, :, k])
                ell_cv[k] += self._cv_fn(self.deltaX[i, :, k], deltaY)
        sample_cov = np.cov(np.exp(ell), ell_cv)
        cv_alpha = sample_cov[0, 1] / sample_cov[1, 1]
        cv_mean = sum(self._cv_mean_fn(np.diff(self.Y)[i]) for i in range(self.m))
        Ell_hat = np.log(np.mean(np.exp(ell) - cv_alpha * (ell_cv - cv_mean)))
        Ell_var = (sample_cov[0, 0] - (sample_cov[0, 1] ** 2 / sample_cov[1, 1])) / self.N
        Ell_re = np.sqrt(Ell_var) / np.exp(Ell_hat)
        return Ell_hat, Ell_var, Ell_re

    def _run_product_cvs(self):
        """
        Runs the product algorithm with control variables and calculates various output statistics.

        Returns:
            Ell_hat (float): MC estimate.
            Ell_var (float): MC variance.
            Ell_re (float): MC relative error.
        """
        ell_hat = np.zeros(self.m)
        ell_hat_var = np.zeros(self.m)
        for i in range(self.m):
            ell = np.zeros(self.N)
            ell_cv = np.zeros(self.N)
            for k in range(self.N):
                self.Y[i, 0] = self.A[i, k]
                deltaY = np.diff(self.Y[i, :])
                ell[k] = self._mvn_pdf_log(deltaY, self.deltaX[i, :, k])
                ell_cv[k] = self._cv_fn(self.deltaX[i, :, k], deltaY)
            sample_cov = np.cov(np.exp(ell), ell_cv)
            cv_alpha = sample_cov[0, 1] / sample_cov[1, 1]
            cv_mean = self._cv_mean_fn(deltaY)
            ell_hat[i] = np.log(np.mean(np.exp(ell) - cv_alpha * (ell_cv - cv_mean)))
            ell_hat_var[i] = (sample_cov[0, 0] - (sample_cov[0, 1] ** 2 / sample_cov[1, 1])) / self.N
        Ell_hat = np.sum(ell_hat)
        Ell_var = np.prod(ell_hat_var + np.exp(ell_hat) ** 2) - np.prod(np.exp(ell_hat) ** 2)
        Ell_re = np.sqrt(Ell_var) / np.exp(Ell_hat)
        return Ell_hat, Ell_var, Ell_re


def run_suite(run_seed, N, algorithms, case_studies, theta_set, output_fname):
    """
    Runs a suite of experiments based on the provided parameters.

    Parameters:
        run_seed (int): The seed for the random number generator.
        N (dict): A dictionary containing the number of MC samples to use for each dataset.
        algorithms (list): A list of algorithms to run the experiments with
                            - subset of ["Original", "Product", "Original_CVs", "Product_CVs"]
        case_studies (list): A list of case studies to perform experiments on
                            - subset of ["A", "B", "C"].
        theta_set (str): Indicates the model parameters to use for experiments.
                            - either "True" or "All"
        output_fname (str): The filename to save the output results.

    Returns:
        None
    """
    # Check if output file already exists, but still overwrite test results
    if os.path.isfile(output_fname) and not output_fname.startswith("./results/results_test"):
        print("Output file already exists - change name or delete.")
        exit()

    data_seed, datasets, thetas = get_parameters(case_studies, theta_set)

    # Run experiments
    results = []
    for dataset in datasets:
        t, n, m = dataset["params"]
        Y = dataset["data"]
        for theta in tqdm(thetas[dataset["name"]]):
            cmc2 = CMC2Algorithm(Y, t, n, m, N[dataset["name"]], theta, run_seed)
            for algorithm in algorithms:
                result = cmc2.run(algorithm)
                results.append([dataset["name"], algorithm, data_seed, run_seed,
                                N[dataset["name"]]] + theta + result)

    # Output results
    results_df = pd.DataFrame(results, columns=["dataset", "algorithm", "data_seed", "run_seed", "N",
                                                "alpha", "eta", "beta", "mu_A", "sigma_A", "sigma_Z",
                                                "ell_hat", "var", "re", "runtime", "rtvp"])
    results_df.to_csv(output_fname, index=False)


if __name__ == "__main__":
    # Experimental suite parameters:
    run_seed = 5
    N = dict({"A": 50000, "B": 50000, "C": 50000})
    algorithms = ["Original", "Product", "Original_CVs", "Product_CVs"]
    case_studies = ["A", "B", "C"]
    theta_set = "True"
    output_fname = "./results/results_test.csv"

    # Run experimental suite
    run_suite(run_seed, N, algorithms, case_studies, theta_set, output_fname)
