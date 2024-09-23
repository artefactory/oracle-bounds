from __future__ import annotations

import pandas as pd
from abc import ABC, abstractmethod


class AForecastModel(ABC):

    def __init__(self, name: str):
        self.name: str = name

    @abstractmethod
    def fit(self, X: pd.DataFrame, Y: pd.DataFrame) -> AForecastModel:
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        pass
