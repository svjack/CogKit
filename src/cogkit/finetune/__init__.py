# -*- coding: utf-8 -*-


from cogkit.finetune.base import BaseTrainer

# import resgistered models
from cogkit.finetune.diffusion import models as diffusion_models
from cogkit.finetune.llm import models as llm_models
from cogkit.finetune.register import get_model_cls, register, show_supported_models
