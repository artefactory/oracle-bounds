import numpy as np
import pandas as pd

from lib.dataprocessor.ADataProcessor import ADataProcessor


class ArDataProcessor(ADataProcessor):
    def __init__(self, name: str, degree: int):
        super().__init__(name)
        self.degree = degree

    def process(self, series):
        X = pd.DataFrame()
        Y = pd.DataFrame(series)
        for i in range(self.degree):
            X[f'x_{i}'] = Y.shift(i+1)
        return np.array(X.values)[self.degree:], np.array(Y.values)[self.degree:]
