from abc import ABC, abstractmethod


class ALoss(ABC):
    @abstractmethod
    def compute(self, expected: float, predicted: float) -> float:
        pass
