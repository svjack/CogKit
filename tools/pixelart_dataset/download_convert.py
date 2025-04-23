import os
import json
from datasets import load_dataset


def generate_sequential_filename(index):
    """Generate sequential filename with leading zeros (00001.png, 00002.png, etc.)."""
    return f"{index + 1:05d}.png"


def setup_directories(base_dir):
    """Create train directory if it doesn't exist."""
    train_dir = os.path.join(base_dir, "train")
    os.makedirs(train_dir, exist_ok=True)
    return train_dir


def main():
    # Setup directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = setup_directories(base_dir)

    # Load the dataset
    print("Loading diffusiondb-pixelart dataset (2k_all subset)...")
    dataset = load_dataset("jainr3/diffusiondb-pixelart", "2k_all")

    # Get all data
    data = dataset["train"]
    num_samples = len(data)

    # Process all data as train data
    train_metadata = []
    print(f"Processing all {num_samples} samples for training...")

    for idx in range(num_samples):
        item = data[idx]
        prompt = item["text"]
        image = item["image"]

        filename = generate_sequential_filename(idx)
        save_path = os.path.join(train_dir, filename)

        # Save the image
        image.save(save_path)

        train_metadata.append({"file_name": filename, "prompt": prompt})

    # Save train metadata
    with open(os.path.join(train_dir, "metadata.jsonl"), "w", encoding="utf-8") as f:
        for item in train_metadata:
            f.write(json.dumps(item) + "\n")

    print("Conversion complete!")
    print(f"Total samples: {len(train_metadata)}")


if __name__ == "__main__":
    main()
