from __future__ import annotations
from typing import Type, TypeVar
from dataclasses import dataclass, fields, MISSING, is_dataclass
import json

from utils.gui_utils import UserPrompt
from utils.io_utils import pickle_load_object, pickle_save_object

T = TypeVar("T", bound="ConfigBase")


@dataclass
class ConfigBase:
    @classmethod
    def load_json(cls: type[T], path: str = None) -> T:
        """
        Load the class from a JSON file.

        Args:
            path (str): The path to the JSON file.

        Returns:
            ConfigBase: The class loaded from the JSON file.

        """
        if path is None:
            path = UserPrompt.open_file(
                title=f"Open {cls.__name__} File",
                file_types=[("json", ".json")],
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
            path = UserPrompt.save_file(
                title=f"Save {type(self).__name__} As",
                file_types=[("json", ".json")],
                defaultextension=".json",
            )

        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    @classmethod
    def load_pickle(cls: type[T], path: str = None) -> T:
        """
        Load the class from a pickle file.

        Args:
            path (str): The path to the pickle file.

        Returns:
            The class loaded from the pickle file.
        """
        if path is None:
            path = UserPrompt.open_file(
                title=f"Open {cls.__name__} File",
                file_types=[("pickle", ".pkl")],
            )

        return pickle_load_object(path)

    def save_pickle(self, path: str = None) -> None:
        """
        Saves the class as a pickle file.

        Args:
            path (str): The path to the output pickle file.
        """
        if path is None:
            path = UserPrompt.save_file(
                title=f"Save {type(self).__name__} As",
                file_types=[("pickle", ".pkl")],
                defaultextension=".pkl",
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
