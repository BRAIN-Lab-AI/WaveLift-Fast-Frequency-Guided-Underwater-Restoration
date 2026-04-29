"""Download/export the UIEB Hugging Face dataset into WaveFlow folder layout.

Expected output:
  data/UIEB/train/input
  data/UIEB/train/target
  data/UIEB/test/input
  data/UIEB/test/target

Usage:
  python scripts/download_uieb_hf.py
"""

from argparse import ArgumentParser
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def save_split(dataset, hf_split: str, local_split: str, root: Path) -> None:
    """Save one HF split with raw/gt fields to input/target folders."""
    input_dir = root / local_split / "input"
    target_dir = root / local_split / "target"
    input_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    for index, example in enumerate(tqdm(dataset[hf_split], desc=f"{hf_split} -> {local_split}")):
        filename = f"{index:04d}.png"
        example["raw"].save(input_dir / filename)
        example["gt"].save(target_dir / filename)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="Hikari0608/UIEB", help="Hugging Face dataset id")
    parser.add_argument("--output", default="data/UIEB", help="Output UIEB root folder")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    root = Path(args.output)

    if "train" not in dataset:
        raise RuntimeError(f"Expected split 'train', found: {list(dataset.keys())}")

    test_split = "val" if "val" in dataset else "validation" if "validation" in dataset else None
    if test_split is None:
        raise RuntimeError(f"Expected split 'val' or 'validation', found: {list(dataset.keys())}")

    save_split(dataset, "train", "train", root)
    save_split(dataset, test_split, "test", root)

    train_count = len(list((root / "train" / "input").glob("*")))
    test_count = len(list((root / "test" / "input").glob("*")))
    print(f"Done. Saved {train_count} train pairs and {test_count} test pairs under {root}.")


if __name__ == "__main__":
    main()
