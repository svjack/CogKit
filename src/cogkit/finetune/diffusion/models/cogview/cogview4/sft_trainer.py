<<<<<<< HEAD
from cogmodels.finetune import register
from cogmodels.finetune.diffusion.models.cogview.cogview4.lora_trainer import Cogview4Trainer
=======
from cogkit.finetune import register
from cogkit.finetune.diffusion.models.cogview.cogview4.lora_trainer import Cogview4Trainer
>>>>>>> test/main


class Cogview4SFTTrainer(Cogview4Trainer):
    pass


register("cogview4-6b", "sft", Cogview4SFTTrainer)
