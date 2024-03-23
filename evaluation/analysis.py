import csv
import numpy as np
import pandas as pd


# TODO: IMPLEMENT LOG PLOTTING HERE INSTEAD IN MAIN.
class Plotter:
    def __init__(self, log_path: str):
        self._data = pd.read_csv(log_path)
        self._header = self._data.columns
        print(self._header)