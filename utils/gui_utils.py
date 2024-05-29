import tkinter as tk
from tkinter import filedialog
from dataclasses import dataclass, fields


# File for GUI utilities, still in progress and experimental


class FocusedWindow:
    def __init__(self):
        root = tk.Tk()
        self.root = root
        self.hide()

    def __enter__(self) -> tk.Tk:
        return self.focus()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.root.destroy()

    def focus(self) -> tk.Tk:
        root = self.root
        root.eval("tk::PlaceWindow %s center" % root.winfo_pathname(root.winfo_id()))
        root.deiconify()
        root.lift()
        root.focus_force()
        return root

    def hide(self) -> tk.Tk:
        root = self.root
        root.withdraw()
        root.overrideredirect(True)
        root.geometry("0x0+0+0")
        return root

    def close(self):
        self.root.destroy()


class UserPrompt:
    @staticmethod
    def open_file(
        title: str = None,
        file_types: list[tuple[str, str]] = None,
        multiple: bool = False,
        **kwargs,
    ) -> str | list[str]:
        if file_types is None:
            file_types = []

        file_types += [("all files", "*.*")]

        with FocusedWindow() as root:
            if multiple:
                path = filedialog.askopenfilenames(
                    parent=root,
                    title=title,
                    filetypes=file_types,
                    **kwargs,
                )
                return list(path)
            else:
                return filedialog.askopenfilename(
                    parent=root,
                    title=title,
                    filetypes=file_types,
                    **kwargs,
                )

    @staticmethod
    def save_file(title: str = None, file_types: list[tuple[str, str]] = None, **kwargs) -> str:
        if file_types is None:
            file_types = []

        file_types += [("all files", "*.*")]

        with FocusedWindow() as root:
            return filedialog.asksaveasfilename(
                parent=root,
                title=title,
                filetypes=file_types,
                confirmoverwrite=True,
                **kwargs,
            )

    @staticmethod
    def open_directory(title: str = None, **kwargs) -> str:
        with FocusedWindow() as root:
            return filedialog.askdirectory(
                parent=root,
                title=title,
                mustexist=True,
                **kwargs,
            )


@dataclass
class ExampleDataClass:
    param1: str
    param2: list[int]
    param3: float


class Config_GUI:
    def __init__(self, data_class: dataclass):
        self.root = tk.Tk()
        self.data_class = data_class
        self.entries = {}

    def create_gui(self):
        for field in fields(self.data_class):
            if field.type not in [int, float, str, list[int]]:
                tk.Button(self.root, text=f"Select {field.name}").pack()
                raise ValueError(f"Type {field.type.__name__} not supported")
            label = tk.Label(self.root, text=f"{field.name}: {field.type.__name__}")
            entry = tk.Entry(self.root)
            label.pack()
            entry.pack()
            self.entries[field.name] = entry

        button = tk.Button(self.root, text="Print values", command=self.print_values)
        button.pack()

        self.root.mainloop()

    def print_values(self):
        for field in fields(self.data_class):
            # entries[field.name] = field.type(entry.get())
            print(field.type)
            print(f"{field.name}: {field.type(self.entries[field.name].get())}")


# Config_GUI(ExampleDataClass).create_gui()
