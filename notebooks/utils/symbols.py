from typing import List, Callable

import numpy as np
import matplotlib.pyplot as plt

base_alphabet = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


class Symbols:
    def __init__(self, data, n_sym, alphabet=None, test="std"):
        self.data = data
        self.n_sym = n_sym
        self.symbolic = []
        self.alphabet = alphabet if alphabet else base_alphabet[:n_sym]
        self.sym_test = self.get_test(test)
        self.sigma = np.std(np.concatenate(self.data))

    def make_symbols(self):
        """
        Create the symbolic series
        :return: None
        """
        for data in self.data:
            idx_array = self.sym_test(data)
            self.symbolic.append([self.alphabet[idx-1] for idx in idx_array])

    def get_test(self, name: str):
        if name is None or name == "std":
            return self.std_test
        elif name == "quantile":
            return self.quantile_test

    def std_test(self, data):
        idx_array = np.zeros(len(data))
        for k in range(self.n_sym - 1):
            idx_array[np.where((k * self.sigma < data) & (data < (k + 1) * self.sigma))] = k + 1
        idx_array[np.where(data > self.n_sym * self.sigma)] = self.n_sym
        return idx_array.astype(int)

    def quantile_test(self, data):
        quantiles = np.linspace(0, 1, self.n_sym + 2)[1:-1]
        sep = np.quantile(data, quantiles)
        idx_array = np.zeros(len(data))+3
        for val in sep:

