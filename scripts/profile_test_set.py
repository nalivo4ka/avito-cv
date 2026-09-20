from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image
from rich.progress import Progress

from avitocv.data.matching.profile import CropProfile
from avitocv.data.matching.profiling import CropMeasurer, CropSetProfiler


def select_paths(images_dir: Path, sample_size: int, seed: int) -> list[Path]:
    paths = sorted(images_dir.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"no png files found in {images_dir}")
    if sample_size <= 0 or sample_size >= len(paths):
        return paths
    rng = np.random.default_rng(seed)
    return [paths[index] for index in sorted(rng.choice(len(paths), size=sample_size, replace=False))]


def iterate_images(paths: list[Path]) -> Iterator[np.ndarray]:
    with Progress() as progress:
        task = progress.add_task("profiling crops", total=len(paths))
        for path in paths:
            with Image.open(path) as image:
                yield np.asarray(image.convert("RGB"), dtype=np.uint8)
            progress.advance(task)


def describe(profile: CropProfile) -> str:
    return (
        f"crops {profile.sample_size}\n"
        f"  height   median {profile.crop_height.median:7.2f}\n"
        f"  aspect   median {profile.aspect_ratio.median:7.2f}\n"
        f"  sharp    median {profile.sharpness.median:7.5f}\n"
        f"  contrast median {profile.ink_spread.median:7.4f}\n"
        f"  grayscale share {profile.grayscale_share:7.3f}"
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure the geometry and quality profile of the test crops")
    parser.add_argument("--images-dir", type=Path, default=Path("test/images"))
    parser.add_argument("--output", type=Path, default=Path("configs/test_profile.json"))
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    paths = select_paths(arguments.images_dir, arguments.sample_size, arguments.seed)
    profile = CropSetProfiler(CropMeasurer()).build(iterate_images(paths))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    profile.save(arguments.output)
    print(f"wrote {arguments.output}")
    print(describe(profile))


if __name__ == "__main__":
    main()
