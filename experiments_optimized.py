import os
import time

import numpy as np
import pandas as pd
from scipy.stats import gamma
from tqdm import tqdm

TRUE_THETAS = {"A": [4, 1, 0.015, 0.0, 0.0, 0.1],
               "B": [2, 2.5, 0.01, 0.5, 0.1, 0.1],
               "C": [4.4, 1.07, 0.04, -4.5, 0.2, 1.3]}  # Approx.
THETA_RANGES = {"A": [np.linspace(3.000, 5.000, 5),
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
                      np.linspace(1.000, 1.600, 5)]}


def load_data(case, fp):
    df = pd.read_csv(fp)
    Y = df.values
    t = [0] + [int(i) for i in df.columns]
    m, n = Y.shape
    if case == "C":
        Y = 100 - Y[:, 1:]
        t = t[1:]
        n -= 1
    return [t, n, m], Y


def get_parameters(case_studies, theta_set):
    # Data seed
    with open("./data/data_seed.txt", "r") as f:
        data_seed = int(f.readline())

    # Datasets
    datasets = []
    for case in case_studies:
        params, data = load_data(case, f"./data/case_{case}.csv")
        datasets.append({"name": case, "params": params, "data": data})

    # Thetas
    thetas = {case: [TRUE_THETAS[case].copy()] for case in case_studies}
    if theta_set == "All":
        for case in case_studies:
            for i, param_range in enumerate(THETA_RANGES[case]):
                for param in param_range:
                    theta_temp = TRUE_THETAS[case].copy()
                    theta_temp[i] = round(param, 5)
                    if theta_temp not in thetas[case]:
                        thetas[case].append(theta_temp)
    elif theta_set != "True":
        raise ValueError("Theta set not recognized")

    return data_seed, datasets, thetas


class CMC2Algorithm:
    def __init__(self, Y, t, n, m, N, theta, run_seed):
        start_time = time.perf_counter()
        rng = np.random.default_rng(run_seed)
        self.m = m
        self.N = N
        self.theta = theta

        dY0 = Y[:, 0][:, np.newaxis] - rng.normal(theta[3], theta[4], size=(m, N))
        dYrest = np.diff(Y, axis=1)
        self.deltaY = np.concatenate((dY0[:, np.newaxis, :],
                                      np.repeat(dYrest[:, :, np.newaxis], N, axis=2)), axis=1)

        self.alphas = np.array([theta[0] * t[j] ** theta[1] - theta[0] * t[j - 1] ** theta[1]
                                for j in range(1, n + 1)])
        deltaX = np.array([[rng.gamma(self.alphas[j], theta[2], size=N) for j in range(n)]
                           for _ in range(m)])

        self.dYdX_diff = self.deltaY - deltaX

        Sigma_Z = np.eye(n) * 2
        Sigma_Z[0, 1] = Sigma_Z[n - 1, n - 2] = - 1
        for i in range(1, n - 1):
            Sigma_Z[i, i - 1] = Sigma_Z[i, i + 1] = - 1
        Sigma_Z *= theta[5] ** 2

        self.mvn_precalc = - 0.5 * (n * np.log(2 * np.pi) + np.log(np.linalg.det(Sigma_Z)) +
                                    np.einsum('ij,njk,nik->kn',
                                              np.linalg.inv(Sigma_Z),
                                              self.dYdX_diff, self.dYdX_diff))

        self.setuptime = time.perf_counter() - start_time

    def run(self, algorithm):
        if algorithm == "Original":
            start_time = time.perf_counter()
            ell_hat, var, re = self._run_original()
        elif algorithm == "Product":
            start_time = time.perf_counter()
            ell_hat, var, re = self._run_product()
        elif algorithm == "Product_CVs":
            start_time = time.perf_counter()
            ell_hat, var, re = self._run_product_cvs()
        else:
            raise ValueError("Algorithm not recognized")
        runtime = (time.perf_counter() - start_time) + self.setuptime
        rtvp = re ** 2 * runtime
        return [ell_hat, var, re, runtime, rtvp]

    def _run_original(self):
        ell = np.sum(self.mvn_precalc, axis=1)
        Ell_hat = - np.log(self.N) + np.max(ell) + np.log(np.sum(np.exp(ell - np.max(ell))))
        Ell_var = np.var(np.exp(ell)) / self.N
        Ell_re = np.sqrt(Ell_var) / np.exp(Ell_hat)
        return Ell_hat, Ell_var, Ell_re

    def _run_product(self):
        ell_hat = np.zeros(self.m)
        ell_hat_var = np.zeros(self.m)
        for i in range(self.m):
            ell = self.mvn_precalc[:, i]
            ell_hat[i] = - np.log(self.N) + np.max(ell) + np.log(np.sum(np.exp(ell - np.max(ell))))
            ell_hat_var[i] = np.var(np.exp(ell)) / self.N
        Ell_hat = np.sum(ell_hat)
        Ell_var = np.prod(ell_hat_var + np.exp(ell_hat) ** 2) - np.prod(np.exp(ell_hat) ** 2)
        Ell_re = np.sqrt(Ell_var) / np.exp(Ell_hat)
        return Ell_hat, Ell_var, Ell_re

    def _run_product_cvs(self):
        ell_hat = np.zeros(self.m)
        ell_hat_var = np.zeros(self.m)
        cv_mean = np.sum(self.alphas[1:] * self.theta[2] - self.deltaY[:, 1:, 0]
                         - 2 * self.alphas[1:] * self.theta[2] *
                         gamma.cdf(self.deltaY[:, 1:, 0], a=self.alphas[1:] + 1, scale=self.theta[2])
                         + 2 * self.deltaY[:, 1:, 0] *
                         gamma.cdf(self.deltaY[:, 1:, 0], a=self.alphas[1:], scale=self.theta[2]),
                         axis=1)
        for i in range(self.m):
            ell = self.mvn_precalc[:, i]
            ell_cv = np.sum(abs(self.dYdX_diff[i, 1:, :]), axis=0)
            sample_cov = np.cov(np.exp(ell), ell_cv)
            cv_alpha = sample_cov[0, 1] / sample_cov[1, 1]
            ell_hat[i] = np.log(np.mean(np.exp(ell) - cv_alpha * (ell_cv - cv_mean[i])))
            ell_hat_var[i] = (sample_cov[0, 0] - (sample_cov[0, 1] ** 2 / sample_cov[1, 1])) / self.N
        Ell_hat = np.sum(ell_hat)
        Ell_var = np.prod(ell_hat_var + np.exp(ell_hat) ** 2) - np.prod(np.exp(ell_hat) ** 2)
        Ell_re = np.sqrt(Ell_var) / np.exp(Ell_hat)
        return Ell_hat, Ell_var, Ell_re


def run_suite(run_seed, N, algorithms, case_studies, theta_set, output_fname):
    if os.path.isfile(output_fname) and not output_fname.startswith("./results/results_test"):
        raise FileExistsError("Output file already exists - change name or delete.")

    data_seed, datasets, thetas = get_parameters(case_studies, theta_set)

    # Run experiments
    results = []
    for dataset in datasets:
        t, n, m = dataset["params"]
        Y = dataset["data"]
        case = dataset["name"]
        for theta in tqdm(thetas[case]):
            cmc2 = CMC2Algorithm(Y, t, n, m, N[case], theta, run_seed)
            for algorithm in algorithms:
                result = cmc2.run(algorithm)
                results.append([case, algorithm, data_seed, run_seed,
                                N[case]] + theta + result)

    # Output results
    results_df = pd.DataFrame(results, columns=["dataset", "algorithm", "data_seed", "run_seed", "N",
                                                "alpha", "eta", "beta", "mu_A", "sigma_A", "sigma_Z",
                                                "ell_hat", "var", "re", "runtime", "rtvp"])
    results_df.to_csv(output_fname, index=False)


if __name__ == "__main__":
    run_seed = 5
    N = dict({"A": 50000, "B": 50000, "C": 50000})
    algorithms = ["Original", "Product", "Product_CVs"]
    case_studies = ["A", "B", "C"]
    theta_set = "All"  # "True" or "All"
    output_fname = "./results/results_test.csv"

    run_suite(run_seed, N, algorithms, case_studies, theta_set, output_fname)
