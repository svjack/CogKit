import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

<<<<<<< HEAD
from cogmodels.finetune import get_model_cls
=======
from cogkit.finetune import get_model_cls
>>>>>>> test/main


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--training_type", type=str, required=True)
    args, unknown = parser.parse_known_args()

    trainer_cls = get_model_cls(args.model_name, args.training_type)
    trainer = trainer_cls()
    trainer.fit()


if __name__ == "__main__":
    main()
