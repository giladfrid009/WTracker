from pathlib import Path, PurePath
import pickle


from pathlib import PurePath

def join_paths(*path_segments: str):
    """
    Join multiple path segments into a single path.

    Args:
        *path_segments: Variable number of path segments to be joined.

    Returns:
        str: The joined path as a string.

    Example:
        >>> join_paths('home', 'yashlat', 'source', 'Bio-Proj', 'data')
        'home/yashlat/source/Bio-Proj/data'
    """
    return PurePath(*path_segments).as_posix()


from pathlib import Path

def create_parent_directory(file_path: str):
    """
    Create the parent directory for the given file path if it doesn't exist.

    Args:
        file_path (str): The path of the file.

    Returns:
        None
    """
    save_folder = Path(file_path).parent
    save_folder.mkdir(parents=True, exist_ok=True)


from pathlib import Path

def create_directory(dir_path: str):
    """
    Create a directory at the specified path if it doesn't already exist.

    Args:
        dir_path (str): The path of the directory to be created.

    Returns:
        None
    """
    Path(dir_path).mkdir(parents=True, exist_ok=True)


import pickle

def pickle_save_object(obj, file_path: str):
    """
    Save an object to a pickle file.

    Args:
        obj: The object to be saved.
        file_path (str): The path to the pickle file.

    Raises:
        ValueError: If there is an error saving the object to the pickle file.
    """
    try:
        create_parent_directory(file_path)
        with open(file_path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        raise ValueError(f"error saving object to pickle file: {e}")


def pickle_load_object(file_path: str):
    """
    Load an object from a pickle file.

    Args:
        file_path (str): The path to the pickle file.

    Returns:
        The loaded object.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If there is an error loading the object from the pickle file.
    """
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"file does not exist: {file_path}")
    except Exception as e:
        raise ValueError(f"error loading object from pickle file: {e}")

import glob
from PIL import Image
import os


def folder_convert_images(folder: str, old_extension: str, new_extension, remove_old: bool = False):
    old_names = glob.glob(f"*.{old_extension}", root_dir=folder)
    new_names = [Path(p).stem + f".{new_extension}" for p in old_names]

    for old_name, new_name in zip(old_names, new_names):
        Image.open(join_paths(folder, old_name)).save(join_paths(folder, new_name))

    if remove_old:
        for old_name in old_names:
            Path(join_paths(folder, old_name)).unlink()


def rename_raw_file_names(folder: str, extension:str):
    for count, filename in enumerate(glob.glob(f"*.{extension}", root_dir=folder)):
        new_name = filename.split('-')[-1]
        # rename all the files
        os.rename(os.path.join(folder, filename),  os.path.join(folder, new_name))
        # print(filename)



