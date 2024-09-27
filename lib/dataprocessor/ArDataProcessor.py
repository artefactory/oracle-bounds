import numpy as np
import pandas as pd

from lib.dataprocessor.ADataProcessor import ADataProcessor


class ArDataProcessor(ADataProcessor):
    def __init__(self, name: str, degree: int):
        super().__init__(name)
        self.degree = degree

    def process(self, series):
        y = pd.DataFrame(series)
        shifted_columns = [y.shift(i + 1).rename(columns={y.columns[0]: f'x_{i}'}) for i in range(self.degree)]
        x = pd.concat(shifted_columns, axis=1)
        return np.array(x.values)[self.degree:], np.array(y.values)[self.degree:]
