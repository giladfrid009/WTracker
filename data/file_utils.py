from pathlib import Path
import pickle


def create_parent_directory(file_path: str):
    save_folder = Path(file_path).parent
    save_folder.mkdir(parents=True, exist_ok=True)


def create_directory(dir_path: str):
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def pickle_save_object(obj, file_path: str):
    try:
        create_parent_directory(file_path)
        with open(file_path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        raise ValueError(f"error saving object to pickle file: {e}")


def pickle_load_object(file_path: str):
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"file does not exist: {file_path}")
    except Exception as e:
        raise ValueError(f"error loading object from pickle file: {e}")
