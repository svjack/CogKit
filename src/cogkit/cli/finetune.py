# -*- coding: utf-8 -*-

# from cogkit.utils.cogvideo.models.utils import get_model_cls
# from cogkit.utils.cogvideo.schemas import Args

# from models.utils import get_model_cls
# from utils.schemas import Args


import click


@click.command()
def finetune() -> None:
    raise NotImplementedError
    args = Args.parse_args()
    trainer_cls = get_model_cls(args.model_name, args.training_type)
    trainer = trainer_cls(args)
    trainer.fit()
