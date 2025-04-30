# CogKit

## Introduction

CogKit is an open-source project that provides a user-friendly interface for researchers and developers to utilize models from ZhipuAI, currently supports [CogView](https://huggingface.co/collections/THUDM/cogview-67ac3f241eefad2af015669b) (image generation) and [CogVideoX](https://huggingface.co/collections/THUDM/cogvideo-66c08e62f1685a3ade464cce) (video generation) series. Users must comply with legal and ethical guidelines to ensure responsible implementation.

Visit our [Docs](https://thudm.github.io/CogKit) to start.

## Features

- Training Optimization: Includes pre-computation and caching of latents and embeddings, sequence packing, and various memory-efficient strategies to improve training throughput and reduce GPU memory usage.

- Native Resolution Training Support: Seamlessly train models at original image resolutions for improved quality and consistency.

- Easy-to-use Interface: Offers multiple easy-to-use inference options, including a CLI, OpenAI-compatible API server, and interactive Gradio-based UIs for both training and inference.

## Roadmap

- [ ] Add support for CogView4 ControlNet model
- [ ] Docker for easy deployment

## License

This project is licensed under the [Apache 2.0 License](LICENSE).
