from abc import abstractmethod


class ADataProcessor:
    def __init__(self, name: str):
        self.name: str = name

    @abstractmethod
    def process(self, data):
        pass
