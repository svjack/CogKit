---
---

# Installation

<<<<<<< HEAD
Cogkit can be installed using pip. We recommend using a virtual environment to avoid conflicts with other packages.

## Requirements

- Python 3.8 or higher
=======
CogKit can be installed using pip. We recommend using a virtual environment to avoid conflicts with other packages.

## Requirements

- Python 3.10 or higher
>>>>>>> test/main
- CUDA-compatible GPU (for optimal performance)
- At least 8GB of GPU memory for inference, 16GB+ recommended for fine-tuning

## Installation Steps

### Create a virtual environment (recommended)

```bash
# Using venv
python -m venv cogkit-env
source cogkit-env/bin/activate

# Or using conda
<<<<<<< HEAD
conda create -n cogkit-env python=3.8
=======
conda create -n cogkit-env python=3.10
>>>>>>> test/main
conda activate cogkit-env
```

### Install PyTorch

Please refer to the [PyTorch installation guide](https://pytorch.org/get-started/locally/) for instructions on installing PyTorch according to your system.

### Install Cogkit
<<<<<<< HEAD
<!-- FIXME: Install via pip install cogkit or via clone&local install? -->

1. Install Cogkit:

   <!-- TODO: add github link -->
   ```bash
   pip install cogkit@git+https:
=======

1. Install Cogkit:
   ```bash
   pip install cogkit@git+https://github.com/thudm/cogkit.git
>>>>>>> test/main
   ```

2. Optional: for video tasks (e.g. text-to-video), install additional dependencies:

   ```bash
   pip install -e .[video]
   ```


### Verify installation

You can verify that cogkit is installed correctly by running:

```bash
<<<<<<< HEAD
python -c "import cogkit"
```

<!-- TODO: add in roadmap -->
## [Optional] Install via docker

If you have any issues with the installation, you can install Cogkit via Docker. We provide a Docker image that includes all dependencies. You can pull the image from Docker Hub:

<!-- FIXME: add link to the docker image -->
```bash
docker pull ghcr.io/cogmodels/cogkit:latest
```

To run the container, use the following command:

<!-- FIXME: add link to the docker image -->
```bash
docker run -it ghcr.io/cogmodels/cogkit:latest
```

## Troubleshooting

For more detailed troubleshooting, please refer to our GitHub issues page.
=======
cogkit --help
```

and will get:

```text
Usage: cogkit [OPTIONS] COMMAND [ARGS]...

Options:
  -v, --verbose  Verbosity level (from 0 to 2)  [default: 0; 0<=x<=2]
  --help         Show this message and exit.

Commands:
  finetune
  inference  Generates a video based on the given prompt and saves it to...
  launch
```
>>>>>>> test/main
