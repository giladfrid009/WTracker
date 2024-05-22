from __future__ import annotations
from dataclasses import dataclass, fields, MISSING
import json
import tkinter as tk
from tkinter import filedialog


@dataclass
class ConfigBase:
    @classmethod
    def load_json(cls, path: str = None) -> ConfigBase:
        """
        Load the class from a JSON file.

        Args:
            path (str): The path to the JSON file.

        Returns:
            ConfigBase: The class loaded from the JSON file.

        """
        if path is None:
            path = filedialog.askopenfilename(
                title=f"open {cls.__name__} file",
                filetypes=[("json", ".json"), ("Any type", ".*")],
            )

        with open(path, "r") as f:
            data = json.load(f)
        obj = cls.__new__(cls)
        obj.__dict__.update(data)
        return obj

    def save_json(self, path: str = None):
        """
        Saves the class as JSON file.

        Args:
            path (str): The path to the output JSON file.
        """
        print(type(self))
        if path is None:
            path = filedialog.asksaveasfilename(
                title=f"save {type(self).__name__} config as",
                defaultextension=".json",
                filetypes=[("json", ".json"), ("Any type", ".*")],
            )
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    @classmethod
    def print_initialization(cls, include_default:bool=True, init_fields_only:bool=True) -> str:
        """
        Print the initialization of the class as a string
        """
        print(f"{cls.__name__}(")
        for field in fields(cls):
            if init_fields_only and field.init is False:
                continue
            
            is_default = not isinstance(field.default, type(MISSING))
            val = None
            if include_default and is_default:
                val = field.default

            if type(val) is str:
                val = f'f"{val}"'
            print(f"    {field.name} = {val}, # {field.type.__name__}")
        print(")")





