# Fine-tuning Dataset Format Guide

This directory contains datasets for fine-tuning different models, divided into three types:

## 1. Image-to-Video Conversion Dataset (i2v)

**Directory Structure**: `i2v/train/` and `i2v/test/`

**Data Format**:
- Each directory contains video files (`.mp4`) and optional corresponding image files (`.png`)
- The `metadata.jsonl` file contains metadata information for each sample

**metadata.jsonl Format**:
```json
{"file_name": "example.mp4", "id": 0, "prompt": "Detailed video description text..."}
{"file_name": "example.png", "id": 0}
```

**Notes**:
- Each video file has a corresponding detailed text description (prompt)
- Image files are optional; if not provided, the system will default to using the first frame of the video as the input image
- When image files are provided, they are associated with the video file of the same name through the id field
- During training, the model will learn to generate corresponding videos from input images

## 2. Text-to-Video Conversion Dataset (t2v)

**Directory Structure**: `t2v/train/` and `t2v/test/`

**Data Format**:
- Each directory contains a set of video files (`.mp4`)
- The `metadata.jsonl` file contains text descriptions for each video

**metadata.jsonl Format**:
```json
{"file_name": "example.mp4", "prompt": "Detailed video description text..."}
```

**Notes**:
- Each video file has a corresponding detailed text description (prompt)
- During training, the model will learn to generate corresponding video content from text descriptions

## 3. Text-to-Image Conversion Dataset (t2i)

**Directory Structure**: `t2i/train/` and `t2i/test/`

**Data Format**:
- Each directory contains a set of image files (`.png`)
- The `metadata.jsonl` file contains text descriptions for each image

**metadata.jsonl Format**:
```json
{"file_name": "example.png", "prompt": "Detailed image description text..."}
```

**Notes**:
- Each image file has a corresponding detailed text description (prompt)
- During training, the model will learn to generate corresponding image content from text descriptions

## Dataset Usage Instructions

1. Training sets (`train/`) are used for model training
2. Test sets (`test/`) are used for evaluating model performance
3. Each dataset will generate a `.cache/` directory during training, used to store preprocessed cache data. If the dataset changes, you need to manually delete this directory and retrain.

## Data Examples

- **i2v**: Contains image and video pairs, along with text describing the video content
- **t2v**: Contains videos and corresponding detailed text descriptions
- **t2i**: Contains images and corresponding detailed text descriptions

All datasets follow a similar organizational structure, facilitating unified data loading and processing workflows.
