import tkinter as tk
from tkinter import filedialog


class FocusedWindow:
    def __init__(self):
        root = tk.Tk()
        self.root = root
        self.hide()

    def __enter__(self) -> tk.Tk:
        return self.focus()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.hide()

    def focus(self) -> tk.Tk:
        root = self.root
        root.eval("tk::PlaceWindow %s center" % root.winfo_pathname(root.winfo_id()))
        root.deiconify()
        root.lift()
        root.attributes("-topmost", True)
        root.focus_force()
        root.update()
        root.after_idle(root.attributes, "-topmost", False)
        return root

    def hide(self) -> tk.Tk:
        root = self.root
        root.withdraw()
        root.overrideredirect(True)
        root.geometry("0x0+0+0")
        root.update()
        return root

    def close(self):
        self.root.destroy()

    def __del__(self):
        self.close()


class UserPrompt:
    """Class for creating a user prompt dialogs."""

    @staticmethod
    def open_file(
        title: str = None,
        file_types: list[tuple[str, str]] = None,
        multiple: bool = False,
        **kwargs,
    ) -> str | list[str]:
        """
        Opens a file dialog to select one or multiple files.

        Args:
            title (str, optional): The title of the file dialog window.
            file_types (list[tuple[str, str]], optional): A list of file types to filter the displayed files. Each file type is represented as a tuple of the form (description, pattern).
            multiple (bool, optional): Whether to allow multiple file selection.
            **kwargs: Additional keyword arguments to be passed to the file dialog.

        Returns:
            str | list[str]: The path of the selected file(s). If multiple is True, a list of paths is returned. Otherwise, a single path is returned.

        """
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
        """
        Opens a file dialog to save a file and returns the selected file path.

        Args:
            title (str, optional): The title of the file dialog window.
            file_types (list[tuple[str, str]], optional): A list of file types to filter the displayed files.
                Each file type is represented as a tuple of the form (description, pattern).
            **kwargs: Additional keyword arguments to be passed to the file dialog.

        Returns:
            str: The selected file path.

        """
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
        """
        Opens a dialog box to select a directory.

        Args:
            title (str, optional): The title of the dialog box.
            **kwargs: Additional keyword arguments to be passed to the filedialog.askdirectory function.

        Returns:
            str: The path of the selected directory.

        """
        with FocusedWindow() as root:
            return filedialog.askdirectory(
                parent=root,
                title=title,
                mustexist=True,
                **kwargs,
            )
