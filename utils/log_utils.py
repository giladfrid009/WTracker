from typing import Iterable, Any
import csv


class TextLogger:
    def __init__(self, path: str, mode: str = "w"):
        self.path = path
        self._file = open(self.path, mode)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        if not self._file.closed:
            self._file.flush()
            self._file.close()

    def write(self, string: str):
        assert self._file.writable()
        self._file.write(string)

    def writelines(self, strings: list[str]):
        assert self._file.writable()
        self._file.writelines(strings)

    def flush(self):
        self._file.flush()


class CSVLogger:
    def __init__(self, path: str, col_names: list[str], mode: str = "w+"):
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
        if not self._file.closed:
            self._file.flush()
            self._file.close()

    def _to_dict(self, items: Iterable) -> dict:
        return {k: v for k, v in zip(self.col_names, items)}

    def write(self, row: dict | Iterable):
        assert self._file.writable()

        if not isinstance(row, dict):
            row = self._to_dict(row)

        self._writer.writerow(row)

    def writerows(self, rows: list[dict] | list[Iterable]):
        assert self._file.writable()
        assert len(rows) > 0

        if not isinstance(rows[0], dict):
            rows = [self._to_dict(row) for row in rows]

        self._writer.writerows(rows)

    def flush(self):
        self._file.flush()
