from __future__ import annotations

import io

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from avitocv.data.real.crop_datasets import CropFileWriter, ExtractedCrop, ParquetCropReader

USABLE_SIZE = (120, 30)
TOO_SHORT_SIZE = (40, 6)
TOO_NARROW_SIZE = (30, 30)


def _image(size: tuple[int, int]) -> Image.Image:
    return Image.new("RGB", size, (180, 170, 160))


def _encoded(size: tuple[int, int]) -> bytes:
    buffer = io.BytesIO()
    _image(size).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def parquet_path(tmp_path):
    """Повторяет формат HuggingFace: картинка лежит структурой с полями bytes и path."""
    sizes = [USABLE_SIZE, TOO_SHORT_SIZE, (200, 40)]
    table = pa.table({
        "image": pa.array(
            [{"bytes": _encoded(size), "path": None} for size in sizes],
            type=pa.struct([("bytes", pa.binary()), ("path", pa.string())]),
        ),
        "text": pa.array(["первый", "второй", "третий"]),
    })
    path = tmp_path / "crops.parquet"
    pq.write_table(table, path)
    return path


class TestExtractedCrop:
    def test_wide_crop_is_usable(self) -> None:
        assert ExtractedCrop(_image(USABLE_SIZE), "текст").is_usable

    def test_too_short_crop_is_rejected(self) -> None:
        assert not ExtractedCrop(_image(TOO_SHORT_SIZE), "текст").is_usable

    def test_square_crop_is_rejected(self) -> None:
        assert not ExtractedCrop(_image(TOO_NARROW_SIZE), "текст").is_usable


class TestParquetCropReader:
    def test_every_row_is_decoded_with_its_text(self, parquet_path) -> None:
        crops = list(ParquetCropReader().read(parquet_path))
        assert [crop.image.size for crop in crops] == [USABLE_SIZE, TOO_SHORT_SIZE, (200, 40)]
        assert [crop.text for crop in crops] == ["первый", "второй", "третий"]

    def test_limit_stops_the_reader_early(self, parquet_path) -> None:
        assert len(list(ParquetCropReader().read(parquet_path, limit=2))) == 2


class TestCropFileWriter:
    def test_unusable_crops_are_skipped(self, parquet_path, tmp_path) -> None:
        records = CropFileWriter(tmp_path / "out").write(ParquetCropReader().read(parquet_path), "slice")
        assert len(records) == 2

    def test_box_covers_the_whole_saved_image(self, parquet_path, tmp_path) -> None:
        records = CropFileWriter(tmp_path / "out").write(ParquetCropReader().read(parquet_path), "slice")
        first = records[0]
        assert (first.left, first.top, first.width, first.height) == (0, 0, *USABLE_SIZE)
        assert Image.open(first.image_path).size == USABLE_SIZE

    def test_records_carry_the_slice_name(self, parquet_path, tmp_path) -> None:
        records = CropFileWriter(tmp_path / "out").write(ParquetCropReader().read(parquet_path), "cyrillic_plate")
        assert {record.slice_name for record in records} == {"cyrillic_plate"}
