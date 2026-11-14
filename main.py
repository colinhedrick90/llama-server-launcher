import multiprocessing
import tkinter as tk

from launcher_app import LauncherApp


def main() -> None:
    multiprocessing.freeze_support()
    root = tk.Tk()
    LauncherApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

