from __future__ import annotations
import os
from pathlib import Path, PurePath
from typing import Callable, Union
import shutil


def absolute_path(file_path: str) -> str:
    """
    Get the absolute path of a file.

    Args:
        file_path (str): The path of the file.

    Returns:
        str: The absolute path of the file.
    """
    return Path(file_path).resolve().as_posix()


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
    path: Path = Path(dir_path)
    for file_name in path.iterdir():
        if file_name.is_dir():
            continue

        new_name = path / rename_fn(file_name.name)
        file_name.rename(new_name)


class Files:
    """
    A utility class for working with files in a directory.

    Args:
        directory (str): The directory path to scan for files.
        extension (str, optional): The file extension to filter the files.
        scan_dirs (bool, optional): Whether to include directories in the results.
        return_full_path (bool, optional): Whether to return the full path of the files.
        sorting_key (Callable[[str], Union[int, str]], optional): A function to determine the sorting order of the files.
    """

    def __init__(
        self,
        directory: str,
        extension: str = "",
        scan_dirs: bool = False,
        return_full_path: bool = True,
        sorting_key: Callable[[str], Union[int, str]] = lambda name: name,
    ) -> None:
        self.root = directory
        self.extension = extension.lower()
        self.scan_dirs: bool = scan_dirs
        self.return_full_path = return_full_path
        self.results: list[os.DirEntry] = []
        self.sorting_func = sorting_key

        self._pos = -1

        self._scan()

    def _scan(self):
        self.results = []
        self._pos = -1

        for result in os.scandir(self.root):
            if self.scan_dirs and result.is_dir():
                self.results.append(result)
            else:
                if result.name.lower().endswith(self.extension):
                    self.results.append(result)

        self.results = sorted(self.results, key=lambda f: self.sorting_func(f.name))

    def __getitem__(self, index: int) -> os.DirEntry:
        """
        Returns the file at the specified index.

        Args:
            index (int): The index of the file.

        Returns:
            os.DirEntry: The file at the specified index.
        """
        return self.results[index]

    def __iter__(self) -> Files:
        """
        Returns an iterator object.

        Returns:
            Files: The iterator object.
        """
        self._pos = -1
        return self

    def __next__(self) -> str:
        """
        Returns the next file name or path in the iteration.

        Returns:
            str: The next file name or path.

        Raises:
            StopIteration: If there are no more files in the iteration.
        """
        self._pos += 1
        if self._pos >= self.__len__():
            raise StopIteration

        result = self.results[self._pos]
        if self.return_full_path:
            return result.path
        return result.name

    def __len__(self) -> int:
        """
        Returns the number of files in the results list.

        Returns:
            int: The number of files.
        """
        return len(self.results)

    def __contains__(self, key: str) -> bool:
        """
        Checks if a file with the specified name exists in the results list.

        Args:
            key (str): The file name to check.

        Returns:
            bool: True if the file exists, False otherwise.
        """
        for res in self.results:
            if key == res.name:
                return True
        return False

    def get_filename(self) -> str:
        """
        Returns the name of the current file.

        Returns:
            str: The name of the current file.
        """
        return self.results[self._pos].name

    def get_path(self) -> str:
        """
        Returns the path of the current file.

        Returns:
            str: The path of the current file.
        """
        return self.results[self._pos].path

    def seek(self, pos: int) -> str:
        """
        Moves the iterator to the specified position and returns the file name or path.

        Args:
            pos (int): The position to seek to.

        Returns:
            str: The file name or path at the specified position.

        Raises:
            AssertionError: If the specified position is invalid.
        """
        assert 0 <= pos < self.__len__(), "Invalid position"
        self._pos = pos - 1
        return self.__next__()

    def copy(self, dst_root: str) -> None:
        """
        Copies the current file to the specified destination directory.

        Args:
            dst_root (str): The destination directory path.
        """
        shutil.copy2(self.get_path(), dst=dst_root)
