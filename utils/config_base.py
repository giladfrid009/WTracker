from __future__ import annotations
from dataclasses import dataclass
import json


@dataclass
class ConfigBase:
    @classmethod
    def load_json(cls, path: str) -> ConfigBase:
        """
        Load the class from a JSON file.

        Args:
            path (str): The path to the JSON file.

        Returns:
            ConfigBase: The class loaded from the JSON file.
            
        """
        with open(path, "r") as f:
            data = json.load(f)
        obj = cls.__new__(cls)
        obj.__dict__.update(data)
        return obj

    def save_json(self, path: str):
        """
        Saves the class as JSON file.

        Args:
            path (str): The path to the output JSON file.
        """
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=4)