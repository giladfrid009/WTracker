from __future__ import annotations
from dataclasses import dataclass, fields, MISSING, is_dataclass
from tkinter import filedialog
import json

from utils.io_utils import pickle_load_object, pickle_save_object


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
        if path is None:
            path = filedialog.asksaveasfilename(
                title=f"save {type(self).__name__} config as",
                defaultextension=".json",
                filetypes=[("json", ".json"), ("Any type", ".*")],
            )
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    @classmethod
    def load_pickle(cls, path: str = None):
        """
        Load the class from a pickle file.

        Args:
            path (str): The path to the pickle file.

        Returns:
            The class loaded from the pickle file.
        """
        if path is None:
            path = filedialog.askopenfilename(
                title=f"open {cls.__name__} file",
                filetypes=[("pickle", ".pkl"), ("Any type", ".*")],
            )
        return pickle_load_object(path)

    def save_pickle(self, path: str = None) -> None:
        """
        Saves the class as a pickle file.

        Args:
            path (str): The path to the output pickle file.
        """
        if path is None:
            path = filedialog.asksaveasfilename(
                title=f"save {type(self).__name__} config as",
                defaultextension=".pkl",
                filetypes=[("pickle", ".pkl"), ("Any type", ".*")],
            )
        pickle_save_object(self, path)


def print_initialization(cls, include_default: bool = True, init_fields_only: bool = True) -> str:
    """
    Print the initialization of a dataclass as a string
    """
    if not is_dataclass(cls):
        print(f"ERROR::{cls.__name__} is not a dataclass")
        return ""

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
        print(f"    {field.name} = {val}, # {field.type}")
    print(")")
