import os
from pathlib import Path, PurePath
from typing import Callable, Union
import shutil


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


class Files:
    def __init__(
        self,
        directory: str,
        extension: str = "",
        scan_dirs: bool = False,
        return_full_path: bool = True,
        sorting_func:Callable[[str], Union[int, str]] = lambda f: f.name,
    ) -> None:
        self.root = directory
        self.extension = extension
        self.scan_dirs: bool = scan_dirs
        self.return_full_path = return_full_path
        self.results: list[os.DirEntry] = []
        self.sorting_func = sorting_func

        self._pos = -1

        self._scan()

    def _scan(self):
        self.results: list = []
        self._pos = -1

        for result in os.scandir(self.root):
            if self.scan_dirs and result.is_dir():
                self.results.append(result)
            else:
                if result.name.endswith(self.extension):
                    self.results.append(result)

        self.results = sorted(self.results, key=lambda f: self.sorting_func(f.name))

    def __getitem__(self, index: int) -> os.DirEntry:
        return self.results[index]

    def __iter__(self):
        self._pos = -1
        return self

    def __next__(self):
        self._pos += 1
        if self._pos >= self.__len__():
            raise StopIteration

        result: os.DirEntry = self.results[self._pos]
        if self.return_full_path:
            return result.path
        return result.name

    def __len__(self) -> int:
        return len(self.results)
    
    def __contains__(self, key:str) -> bool:
        for res in self.results:
            if key == res.name:
                return True
        return False

    def get_filename(self) -> str:
        return self.results[self._pos].name

    def get_path(self) -> str:
        return self.results[self._pos].path

    def seek(self, pos: int):
        if 0 <= pos < self.__len__():
            self._pos = pos - 1
            return self.__next__()
    
    def copy(self, dst_root:str) -> None:
        shutil.copy2(self.get_path(), dst=dst_root)
    
