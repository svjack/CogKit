from cogkit.finetune import register
from cogkit.finetune.diffusion.models.cogview.cogview4.lora_trainer import Cogview4Trainer


class Cogview4SFTTrainer(Cogview4Trainer):
    pass


register("cogview4-6b", "sft", Cogview4SFTTrainer)
