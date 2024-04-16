from dataclasses import dataclass
import json


@dataclass
class ConfigBase:
    # TODO: detect automatically the class of derived class and create it automatically, instead of passing 
    # the class as parameter
    @staticmethod
    def load_from_json(filepath, obj_type):
        """
        Loads a JSON file and returns the data as a dictionary.

        Args:
            filepath (str): The path to the JSON file.
            obj_type (type): The type of object to create from the JSON data.

        Returns:
            dict: The dictionary containing the data from the JSON file.
        """
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            return obj_type(**data)
        except FileNotFoundError:
            print(f"Error: File not found: {filepath}")
            return None
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON format in file: {filepath}")
            print(f"Error details: {e}")
            return None

    def save_json(self, filepath):
        """
        Saves a dictionary as JSON data to a file.

        Args:
            data (dict): The dictionary to save as JSON.
            filepath (str): The path to the output JSON file.
        """
        try:
            with open(filepath, "w") as f:
                json.dump(self.__dict__, f, indent=4)
        except IOError as e:
            print(f"Error: Failed to save JSON data to file: {filepath}")
            print(f"Error details: {e}")