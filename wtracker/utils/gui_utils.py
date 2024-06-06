import tkinter as tk
import tkfilebrowser

# TODO: REVERT TO OLD GUI UTILS. this library is shit.

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
        root.lift()
        root.eval("tk::PlaceWindow %s center" % root.winfo_pathname(root.winfo_id()))
        root.deiconify()
        root.focus_force()
        root.attributes("-topmost", True)
        root.update_idletasks()
        return root

    def hide(self) -> tk.Tk:
        root = self.root
        root.withdraw()
        root.update_idletasks()
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
            title (str, optional): The title of the file dialog window. Defaults to None.
            file_types (list[tuple[str, str]], optional): A list of file types to filter the displayed files. Each file type is represented as a tuple of the form (description, pattern). Defaults to None.
            multiple (bool, optional): Whether to allow multiple file selection. Defaults to False.
            **kwargs: Additional keyword arguments to be passed to the tkfilebrowser.askopenfilename or tkfilebrowser.askopenfilenames function.

        Returns:
            str | list[str]: The path of the selected file(s). If multiple is True, a list of paths is returned. Otherwise, a single path is returned.

        """
        if file_types is None:
            file_types = []

        file_types += [("all files", "*.*")]

        if title is None:
            title = "Select a file"

        with FocusedWindow() as root:
            if multiple:
                path = tkfilebrowser.askopenfilenames(
                    parent=root,
                    title=title,
                    filetypes=file_types,
                    initialdir=".",
                    **kwargs,
                )
                return list(path)

            return tkfilebrowser.askopenfilename(
                parent=root,
                title=title,
                filetypes=file_types,
                initialdir=".",
                **kwargs,
            )

    @staticmethod
    def save_file(title: str = None, file_types: list[tuple[str, str]] = None, **kwargs) -> str:
        """
        Opens a file dialog to save a file and returns the selected file path.

        Args:
            title (str, optional): The title of the file dialog window. Defaults to None.
            file_types (list[tuple[str, str]], optional): A list of file types to filter the displayed files.
                Each file type is represented as a tuple of the form (description, pattern). Defaults to None.
            **kwargs: Additional keyword arguments to be passed to the tkfilebrowser.asksaveasfilename function.

        Returns:
            str: The selected file path.

        """
        if file_types is None:
            file_types = []

        file_types += [("all files", "*.*")]

        if title is None:
            title = "Save file"

        with FocusedWindow() as root:
            return tkfilebrowser.asksaveasfilename(
                parent=root,
                title=title,
                filetypes=file_types,
                initialdir=".",
                okbuttontext="Save",
                **kwargs,
            )

    @staticmethod
    def open_directory(title: str = None, multiple: bool = False, **kwargs) -> str | list[str]:
        """
        Opens a dialog box to select a directory.

        Args:
            title (str, optional): The title of the dialog box. Defaults to None.
            multiple (bool, optional): Whether to allow multiple directory selection. Defaults to False.
            **kwargs: Additional keyword arguments to be passed to the filedialog.askdirectory function.

        Returns:
            str | list[str]: The path of the selected directory. If multiple is True, a list of paths is returned. Otherwise, a single path is returned.

        """
        if title is None:
            title = "Select a directory"

        with FocusedWindow() as root:
            if multiple:
                path = tkfilebrowser.askopendirnames(
                    parent=root,
                    title=title,
                    initialdir=".",
                    **kwargs,
                )
                return list(path)

            return tkfilebrowser.askopendirname(
                parent=root,
                title=title,
                initialdir=".",
                **kwargs,
            )
