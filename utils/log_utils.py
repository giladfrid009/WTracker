import typing
from dataclasses import dataclass, field
import csv


@dataclass
class TextLogger:
    path: str
    mode: str = field(default="w")
    file: typing.TextIO = field(init=False)

    def __post_init__(self):
        self.file = open(self.path, self.mode)

    def close(self):
        if not self.file.closed():
            self.file.flush()
            self.file.close()
        pass

    def write(self, string: str):
        if self.file.writable():
            self.file.write(string)

    def writelines(self, strings: list[str]):
        if self.file.writable():
            self.file.writelines(strings)

    def flush(self):
        self.file.flush()


@dataclass
class CSVLogger:
    path: str
    col_names: list[str]
    mode: str = field(default="w+")
    file: typing.TextIO = field(init=False)
    writer: csv.DictWriter = field(init=False)

    def __post_init__(self):
        self.file = open(self.path, self.mode, newline="")
        self.writer = csv.DictWriter(self.file, self.col_names, escapechar=",")
        self.writer.writeheader()
        self.flush()

    def close(self):
        if not self.file.closed():
            self.file.flush()
            self.file.close()
        pass

    def write(self, row: dict):
        if self.file.writable():
            self.writer.writerow(row)

    def writerows(self, rows: list[dict]):
        if self.file.writable():
            self.writer.writerows(rows)

    def flush(self):
        self.file.flush()
