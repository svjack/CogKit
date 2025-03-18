# -*- coding: utf-8 -*-


<<<<<<< HEAD
import os
=======
>>>>>>> test/main
from pathlib import Path


def resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def mkdir(dir_path: str | Path) -> Path:
<<<<<<< HEAD
    dir_pth = resolve_path(dir_path)
    if dir_pth.is_file():
        err_msg = f"Path '{os.fspath(dir_path)}' is a regular file."
        raise ValueError(err_msg)

    if not dir_pth.is_dir():
        dir_pth.mkdir(parents=True, exist_ok=False)
    return dir_pth
=======
    abs_path = resolve_path(dir_path)
    if not abs_path.is_dir():
        abs_path.mkdir(parents=True, exist_ok=False)
    return abs_path
>>>>>>> test/main
