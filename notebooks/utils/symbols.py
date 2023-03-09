from typing import List, Callable

import numpy as np
import matplotlib.pyplot as plt

base_alphabet = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


class Symbols:
    def __init__(self, data, n_sym, alphabet=None, test=None):
        self.data = data
        self.symbolic = []
        self.alphabet = alphabet if alphabet else base_alphabet[:n_sym]
        self.sym_test = test if test else self.std_test
        self.sigma = np.std(np.concatenate(self.data))

    def make_symbols(self):
        """
        Create the symbolic series
        :return: None
        """
        for data in self.data:
            idx_array = self.sym_test(data)
            self.symbolic.append([self.alphabet[idx-1] for idx in idx_array])

    def std_test(self, data):
        idx_array = np.zeros(len(data))
        for k in range(len(self.alphabet) - 1):
            idx_array[np.where((k * self.sigma < data) & (data < (k + 1) * self.sigma))] = k + 1
        idx_array[np.where(data > len(self.alphabet) * self.sigma)] = len(self.alphabet)
        return idx_array.astype(int)
