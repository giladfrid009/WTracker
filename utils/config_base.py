from __future__ import annotations
from dataclasses import dataclass
import json
import tkinter as tk
from tkinter import filedialog


@dataclass
class ConfigBase:
    @classmethod
    def load_json(cls, path: str = None, title: str = "open config file") -> ConfigBase:
        """
        Load the class from a JSON file.

        Args:
            path (str): The path to the JSON file.

        Returns:
            ConfigBase: The class loaded from the JSON file.

        """
        if path is None:
            """ path = filedialog.askopenfilename(
                title=title, filetypes=[("json", ".json"), ("Any type", ".*")]
            ) """

            path = filedialog.askopenfilename(
                title=f"open {cls.__name__} file",
                filetypes=[("json", ".json"), ("Any type", ".*")],
            )

        with open(path, "r") as f:
            data = json.load(f)
        obj = cls.__new__(cls)
        obj.__dict__.update(data)
        return obj

    def save_json(self, path: str = None, title: str = "save config as"):
        """
        Saves the class as JSON file.

        Args:
            path (str): The path to the output JSON file.
        """
        print(type(self))
        if path is None:
            """ path = filedialog.asksaveasfilename(
                title=title,
                defaultextension=".json",
                filetypes=[("json", ".json"), ("Any type", ".*")],
            ) """

            path = filedialog.asksaveasfilename(
                title=f"save {type(self).__name__} config as",
                defaultextension=".json",
                filetypes=[("json", ".json"), ("Any type", ".*")],
            )
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=4)
