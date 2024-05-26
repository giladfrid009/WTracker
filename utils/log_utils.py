import csv
from typing import Iterable

class CSVLogger:
    """
    A class for logging data to a CSV file.

    Args:
        path (str): The path to the CSV file.
        col_names (list[str]): The column names for the CSV file.
        mode (str, optional): The file mode to open the CSV file in. Defaults to "w+".

    Attributes:
        path (str): The path to the CSV file.
        col_names (list[str]): The column names for the CSV file.

    Methods:
        __init__: Initializes the CSVLogger object.
        __enter__: Allows the CSVLogger object to be used as a context manager.
        __exit__: Closes the CSV file when exiting the context.
        close: Closes the CSV file.
        _to_dict: Converts an iterable of items to a dictionary using the column names as keys.
        write: Writes a single row of data to the CSV file.
        writerows: Writes multiple rows of data to the CSV file.
        flush: Flushes any buffered data to the CSV file.

    """

    def __init__(self, path: str, col_names: list[str], mode: str = "w+"):
        """
        Initializes the CSVLogger object.

        Args:
            path (str): The path to the CSV file.
            col_names (list[str]): The column names for the CSV file.
            mode (str, optional): The file mode to open the CSV file in. Defaults to "w+".
        """
        self.path = path
        self.col_names = col_names
        self._file = open(self.path, mode, newline="")
        self._writer = csv.DictWriter(self._file, self.col_names, escapechar=",")
        self._writer.writeheader()
        self.flush()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        """
        Closes the CSV file.
        """
        if not self._file.closed:
            self._file.flush()
            self._file.close()

    def _to_dict(self, items: Iterable) -> dict:
        """
        Converts an iterable of items to a dictionary using the column names as keys.

        Args:
            items (Iterable): The items to convert to a dictionary.

        Returns:
            dict: The dictionary with column names as keys and items as values.
        """
        return {k: v for k, v in zip(self.col_names, items)}

    def write(self, row: dict | Iterable):
        """
        Writes a single row of data to the CSV file.

        Args:
            row (dict | Iterable): The row of data to write to the CSV file.
                If a dictionary is provided, the keys should match the column names.
                If an iterable is provided, the items will be matched with the column names in order.
        """
        assert self._file.writable()

        if not isinstance(row, dict):
            row = self._to_dict(row)

        self._writer.writerow(row)

    def writerows(self, rows: list[dict] | list[Iterable]):
        """
        Writes multiple rows of data to the CSV file.

        Args:
            rows (list[dict] | list[Iterable]): The rows of data to write to the CSV file.
                If a list of dictionaries is provided, the keys should match the column names.
                If a list of iterables is provided, the items will be matched with the column names in order.
        """
        assert self._file.writable()
        assert len(rows) > 0

        if not isinstance(rows[0], dict):
            rows = [self._to_dict(row) for row in rows]

        self._writer.writerows(rows)

    def flush(self):
        """
        Flushes any buffered data to the CSV file.
        """
        self._file.flush()
