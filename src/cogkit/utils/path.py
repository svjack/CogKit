# -*- coding: utf-8 -*-


from pathlib import Path


def resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def mkdir(dir_path: str | Path) -> Path:
    abs_path = resolve_path(dir_path)
    if not abs_path.is_dir():
        abs_path.mkdir(parents=True, exist_ok=False)
    return abs_path
