from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from avitocv.data.sampling import SeedScheme
from avitocv.data.sources import MaterializedTextLineSource
from avitocv.data.storage import CropIndex, CropShardReader, CropShardWriter, decode_crop, encode_crop

CROP_COUNT = 40
JPEG_QUALITY = 95


def _make_crop(index: int) -> Image.Image:
    rng = np.random.default_rng(index)
    height = int(rng.integers(12, 60))
    width = int(height * rng.uniform(2.0, 9.0))
    return Image.fromarray(rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8), "RGB")


@pytest.fixture
def crop_store(tmp_path):
    crops = [_make_crop(index) for index in range(CROP_COUNT)]
    records = []
    with CropShardWriter(tmp_path, shard_size_bytes=8 * 1024) as writer:
        for crop in crops:
            location = writer.write(encode_crop(crop, JPEG_QUALITY))
            records.append({
                "shard": location.shard,
                "offset": location.offset,
                "length": location.length,
                "width": crop.width,
                "height": crop.height,
            })
    CropIndex.from_records(records).save(tmp_path)
    return tmp_path, crops


class TestCropSharding:
    def test_every_crop_is_recovered_with_its_geometry(self, crop_store) -> None:
        directory, crops = crop_store
        index = CropIndex.load(directory)
        reader = CropShardReader(directory)
        for position, expected in enumerate(crops):
            decoded = decode_crop(reader.read(index.location_at(position)))
            assert decoded.size == expected.size

    def test_small_shard_size_spreads_crops_over_several_files(self, crop_store) -> None:
        directory, _ = crop_store
        assert CropIndex.load(directory).frame["shard"].nunique() > 1

    def test_reader_survives_pickling_for_dataloader_workers(self, crop_store) -> None:
        import pickle

        directory, crops = crop_store
        index = CropIndex.load(directory)
        reader = pickle.loads(pickle.dumps(CropShardReader(directory)))
        assert decode_crop(reader.read(index.location_at(0))).size == crops[0].size

    def test_index_rejects_missing_columns(self) -> None:
        import pandas as pd

        with pytest.raises(ValueError):
            CropIndex(pd.DataFrame({"shard": [0], "offset": [0]}))


class TestMaterializedSource:
    def test_source_length_matches_the_index(self, crop_store) -> None:
        directory, crops = crop_store
        assert len(MaterializedTextLineSource(directory)) == len(crops)

    def test_loaded_crops_match_the_stored_geometry(self, crop_store) -> None:
        directory, crops = crop_store
        source = MaterializedTextLineSource(directory)
        scheme = SeedScheme(0)
        for position in range(len(crops)):
            assert source.load_upright(position, scheme.rng_for(position)).size == crops[position].size

    def test_missing_index_is_reported(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            MaterializedTextLineSource(tmp_path)
