# -*- coding: utf-8 -*-


from pathlib import Path

import click


@click.command()
@click.option(
    "--task",
)
@click.argument("model_id_or_path")
@click.argument("save_path", type=click.Path(dir_okay=False, writable=True))
def inference(
    model_id_or_path: str, save_path: str | Path, task: None = None
) -> None:
    raise NotImplementedError
