from abc import ABC, abstractmethod
import pandas as pd

class AbstractDataParser(ABC):
    def __init__(self, path):
        self.filepath = path
        self.data = None

    @abstractmethod
    def parse(self):
        pass