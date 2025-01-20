# -*- coding: utf-8 -*-


import logging
from typing import Annotated

import typer

from cogmodels.cli.demo import demo
from cogmodels.cli.finetune import finetune
from cogmodels.cli.inference import inference

_logger = logging.getLogger(__name__)
app = typer.Typer(pretty_exceptions_show_locals=False)


@app.callback()
def main(
    *,
    verbose: Annotated[
        bool, typer.Option('--verbose', '-v', help='enables verbose mode')
    ] = False,
    debug: Annotated[
        bool, typer.Option('--debug', help='enables debug mode')
    ] = False,
) -> None:
    logging.basicConfig(
        format='%(asctime)s [%(levelname)s][%(filename)s:%(lineno)d] %(message)s'
    )
    log_level = logging.WARNING
    if verbose:
        log_level = logging.INFO
        if debug:
            _logger.warning(
                'No need to enable verbose mode since debug mode is enabled.'
            )
    elif debug:
        log_level = logging.DEBUG
    logging.root.setLevel(log_level)


app.command()(demo)
app.command()(finetune)
app.command()(inference)
