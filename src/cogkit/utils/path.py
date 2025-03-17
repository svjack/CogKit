# -*- coding: utf-8 -*-


import os
from pathlib import Path


def resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def mkdir(dir_path: str | Path) -> Path:
    dir_pth = resolve_path(dir_path)
    if dir_pth.is_file():
        err_msg = f"Path '{os.fspath(dir_path)}' is a regular file."
        raise ValueError(err_msg)

    if not dir_pth.is_dir():
        dir_pth.mkdir(parents=True, exist_ok=False)
    return dir_pth
