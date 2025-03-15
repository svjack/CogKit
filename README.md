# CogKit

## Introduction

**CogKit** is an open-source initiative by Zhipu AI that provides a user-friendly interface, enabling researchers and developers to access and manipulate the Cog family of models.
You can check [here](docs/05-Model%20Card.md) to view support models. The project aims to streamline the application of Cog models across multimodal generation tasks such as **text-to-image (t2i)**, **text-to-video (t2v)**, **image-to-video (i2v)**. 
It should be noted that utilization of CogKit and associated Cog models must adhere to relevant legal frameworks and ethical guidelines to ensure responsible and ethical implementation.

## Features

- Multiple models: CogVideoX, CogVideoX1.5, CogView4.
- Ensemble methods: (incremental) pre-training, (multimodal) instruction.
- Multiple precisions: 16-bit full parameter fine-tuning, frozen fine-tuning, LoRA fine-tuning.
- Fine-tuning methods: single machine single card, single machine multiple cards, multiple machines multiple cards.
- Wide range of tasks: multi-round dialogue, image generation, video generation, etc.
- Extreme reasoning: based on OpenAI style API, browser interface and command line interface.
- Embed Cache: Reduce GPU memory usage.

## Roadmap

- [ ] Add support for CogView4 ControlNet model
- [ ] Docker Image for easy deployment

## License

This project is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for more details.
