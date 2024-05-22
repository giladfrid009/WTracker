import tkinter as tk
from tkinter import filedialog
from dataclasses import dataclass, fields


# File for GUI utilities, still in progress and experemental

class UserPrompt:
    root = tk.Tk()
    def __init__(self):
        self.root.withdraw()
        pass
     
    @staticmethod
    def file_path(path:str=None, title:str=None, file_types:tuple[str] = [], **kwargs) -> str:
        if path is None:
            file_types += [("all files", "*.*")]
            path = filedialog.askopenfilename(title=title, filetypes=file_types, **kwargs)
        return path
    
    @staticmethod
    def directory_path(title:str=None, **kwargs) -> str:
        
        return filedialog.askdirectory(title=title, **kwargs)

    @staticmethod
    def save_file_path(title:str=None, file_types:tuple[str] = [], **kwargs) -> str:
        file_types += [("all files", "*.*")]
        return filedialog.asksaveasfilename(title=title, filetypes=file_types, **kwargs)


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
