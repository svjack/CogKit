# -*- coding: utf-8 -*-


from diffusers.loaders import CogVideoXLoraLoaderMixin, CogView4LoraLoaderMixin


def load_lora_checkpoint(
    pipeline: CogVideoXLoraLoaderMixin | CogView4LoraLoaderMixin,
    lora_model_id_or_path: str,
    lora_rank: int,
) -> None:
    pipeline.load_lora_weights(
        lora_model_id_or_path,
        # TODO: ensures the name is correct
        weight_name="pytorch_lora_weights.safetensors",
        adapter_name="test_1",
    )
    pipeline.fuse_lora(components=["transformer"], lora_scale=1 / lora_rank)
