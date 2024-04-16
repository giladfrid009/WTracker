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
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return cls(**data)
        except FileNotFoundError:
            print(f"Error: File not found: {path}")
            return None
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON format in file: {path}")
            print(f"Error details: {e}")
            return None

    def save_json(self, path: str):
        """
        Saves the class as JSON file.

        Args:
            path (str): The path to the output JSON file.
        """
        try:
            with open(path, "w") as f:
                json.dump(self.__dict__, f, indent=4)
        except IOError as e:
            print(f"Error: Failed to save JSON data to file: {path}")
            print(f"Error details: {e}")