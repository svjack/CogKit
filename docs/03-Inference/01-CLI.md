---
sidebar_position: 1
---
<!-- FIXME: change cogmodels to cogkit-->

<!-- TODO: check this doc -->
# Command-Line Interface

CogModels provides a powerful command-line interface (CLI) that allows you to perform various tasks without writing Python code. This guide covers the available commands and their usage.

## Overview

The main CLI command is `cogmodels`, which provides several subcommands:

```bash
cogmodels [OPTIONS] COMMAND [ARGS]...
```

Available commands:
- `inference`: Generate images or videos using AI models
<!-- FIXME: remove this? -->
- `finetune`: Fine-tune models with custom data
- `launch`: Launch a web UI for interactive use

Global options:
- `-v, --verbose`: Increase verbosity (can be used multiple times)

## Inference Command

The `inference` command allows you to generate images and videos:

```bash
cogmodels inference [OPTIONS] PROMPT MODEL_ID_OR_PATH
```

### Examples

```bash
# Generate an image from text
cogmodels inference "a beautiful sunset over mountains" runwayml/stable-diffusion-v1-5 --task t2i

# Generate a video from text
cogmodels inference "a cat playing with a ball" stabilityai/stable-video-diffusion-img2vid --task t2v

# Generate a video from an image
cogmodels inference "extend this image into a video" stabilityai/stable-video-diffusion-img2vid --task i2v --image_file input.png
```

<!-- FIXME: remove this? -->
## Fine-tuning Command

The `finetune` command allows you to fine-tune models with your own data:

```bash
cogmodels finetune [OPTIONS]
```

> Note: The fine-tuning command is currently under development. Please check back for updates.

## Launch Command

The `launch` command starts a web UI for interactive use:

```bash
cogmodels launch [OPTIONS]
```

This launches a web interface where you can:
- Generate images and videos interactively
- Upload images for image-to-video generation
- Adjust generation parameters
- View and download results

### Options

| Option | Description |
|--------|-------------|
| `--host TEXT` | Host to bind the server to (default: 127.0.0.1) |
| `--port INTEGER` | Port to bind the server to (default: 7860) |
| `--share` | Create a public URL |

### Example

```bash
# Launch the web UI on the default port
cogmodels launch

# Launch the web UI with a public URL
cogmodels launch --share
```

## Logging and Debugging

CogModels CLI provides different verbosity levels for logging:

```bash
# Normal output
cogmodels inference "prompt" model_id

# Verbose output (info level)
cogmodels -v inference "prompt" model_id

# Very verbose output (debug level)
cogmodels -vv inference "prompt" model_id
```

## Environment Variables

The CLI behavior can be modified with environment variables:

- `COGMODELS_CACHE_DIR`: Directory to store cached models and data
- `COGMODELS_OFFLINE`: Set to "1" to run in offline mode
- `COGMODELS_VERBOSE`: Set verbosity level (0-2)

Example:
```bash
COGMODELS_CACHE_DIR=/path/to/cache cogmodels inference "prompt" model_id
```
