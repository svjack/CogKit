import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from cogmodel.utils.cogvideo.models.utils import get_model_cls
from cogmodel.utils.cogvideo.schemas import Args


def main():
    args = Args.parse_args()
    trainer_cls = get_model_cls(args.model_name, args.training_type)
    trainer = trainer_cls(args)
    trainer.fit()


if __name__ == "__main__":
    main()
