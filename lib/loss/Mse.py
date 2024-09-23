import pandas as pd

from lib.loss.ALoss import ALoss


class Mse(ALoss):
    def compute(self, expected: float, predicted: float) -> float:
        return (expected - predicted) ** 2

    def compute_from_dataframe(
            self,
            df: pd.DataFrame,
            prediction_col: list,
            target_col: str = 'y',
            oracle_col: str = 'oracle') -> pd.DataFrame:
        loss_df = pd.DataFrame(self.compute(df[target_col], df[oracle_col]), columns=['oracle'])
        for col in prediction_col:
            loss_df[col] = self.compute(df[target_col], df[col])
        return loss_df
