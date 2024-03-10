from pathlib import Path, PurePath
from typing import Callable


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


def create_directory(dir_path: str):
    """
    Create a directory at the specified path if it doesn't already exist.

    Args:
        dir_path (str): The path of the directory to be created.

    Returns:
        None
    """
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def bulk_rename(dir_path: str, rename_fn: Callable[[str], str]):
    """
    Rename all files in a directory using the provided renaming function.

    Args:
        dir_path (str): The path of the directory containing the files to be renamed.
        rename_fn (Callable[[str], str]): The function to be used for renaming the files.

    Returns:
        None
    """
    dir_path: Path = Path(dir_path)
    for file_name in dir_path.iterdir():
        if file_name.is_dir():
            continue

        new_name = dir_path / rename_fn(file_name.name)
        file_name.rename(new_name)
