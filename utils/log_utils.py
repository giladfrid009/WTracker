from typing import Iterable, Any
import csv


class TextLogger:
    def __init__(self, path: str, mode: str = "w"):
        self.path = path
        self.file = open(self.path, mode)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        if not self.file.closed:
            self.file.flush()
            self.file.close()

    def write(self, string: str):
        assert self.file.writable()
        self.file.write(string)

    def writelines(self, strings: list[str]):
        assert self.file.writable()
        self.file.writelines(strings)

    def flush(self):
        self.file.flush()


class CSVLogger:
    def __init__(self, path: str, col_names: list[str], mode: str = "w+"):
        self.path = path
        self.col_names = col_names
        self.file = open(self.path, mode, newline="")
        self.writer = csv.DictWriter(self.file, self.col_names, escapechar=",")
        self.writer.writeheader()
        self.flush()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        if not self.file.closed:
            self.file.flush()
            self.file.close()

    def _to_dict(self, items: Iterable[Any]) -> dict:
        return {k: v for k, v in zip(self.col_names, items)}

    def write(self, row: dict | Iterable[Any]):
        assert self.file.writable()

        if not isinstance(row, dict):
            row = self._to_dict(row)

        self.writer.writerow(row)

    def writerows(self, rows: list[dict]):
        assert self.file.writable()
        assert len(rows) > 0

        if not isinstance(rows[0], dict):
            rows = [self._to_dict(row) for row in rows]

        self.writer.writerows(rows)

    def flush(self):
        self.file.flush()
