# stdlib
from abc import ABCMeta, abstractmethod

# third party
import pandas as pd


class GeneratorInterface(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, data: pd.DataFrame) -> "GeneratorInterface":
        ...

    @abstractmethod
    def generate(self, count: int) -> pd.DataFrame:
        ...
