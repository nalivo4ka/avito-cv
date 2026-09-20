from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import fsspec
import numpy as np
import pyarrow.parquet as pq
from rich.progress import Progress

from avitocv.data.synthesis.writing_systems import Script, ScriptDetector

WIKIPEDIA_URL_TEMPLATE = (
    "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/{config}/train-{shard:05d}-of-{shards:05d}.parquet"
)
HTTP_BLOCK_SIZE = 4 * 1024 * 1024
TEXT_COLUMN = "text"
SENTENCE_SPLIT_PATTERN = re.compile(r"[.!?\n]+")
WHITESPACE_PATTERN = re.compile(r"\s+")
MARKUP_MARKERS = ("|", "{{", "]]", "http", "==", "<", ">")
PUNCTUATION = ".,:;!?()\"'/%#&+*=@-" + "«»—–№₽"
MIN_SCRIPT_SHARE = 0.6
MIN_LETTER_COUNT = 3
MIN_WORDS = 2
MAX_WORDS = 14
MIN_LINE_LENGTH = 4
MAX_LINE_LENGTH = 120


@dataclass(frozen=True)
class CorpusSpec:
    name: str
    config: str
    shard: int
    shard_count: int
    script: Script
    line_target: int

    @property
    def url(self) -> str:
        return WIKIPEDIA_URL_TEMPLATE.format(config=self.config, shard=self.shard, shards=self.shard_count)


class ParquetArticleStream:
    def __init__(self, url: str, column: str = TEXT_COLUMN) -> None:
        self._url = url
        self._column = column

    def iterate(self) -> Iterator[str]:
        handle = fsspec.filesystem("http").open(self._url, block_size=HTTP_BLOCK_SIZE)
        parquet_file = pq.ParquetFile(handle)
        for group_index in range(parquet_file.metadata.num_row_groups):
            table = parquet_file.read_row_group(group_index, columns=[self._column])
            for value in table.column(self._column):
                yield value.as_py()


class LineExtractor:
    def __init__(self, script: Script, detector: ScriptDetector, rng: np.random.Generator) -> None:
        self._script = script
        self._detector = detector
        self._rng = rng
        self._allowed = frozenset(PUNCTUATION + " ")

    def extract(self, article: str) -> Iterator[str]:
        for sentence in SENTENCE_SPLIT_PATTERN.split(article):
            yield from self._split_into_lines(sentence)

    def _split_into_lines(self, sentence: str) -> Iterator[str]:
        cleaned = self._clean(sentence)
        if cleaned is None:
            return
        words = cleaned.split(" ")
        position = 0
        while position < len(words):
            span = int(self._rng.integers(MIN_WORDS, MAX_WORDS + 1))
            line = " ".join(words[position:position + span])
            position += span
            if self._is_usable(line):
                yield line

    def _clean(self, sentence: str) -> str | None:
        if any(marker in sentence for marker in MARKUP_MARKERS):
            return None
        kept = "".join(character for character in sentence if self._is_allowed(character))
        collapsed = WHITESPACE_PATTERN.sub(" ", kept).strip()
        return collapsed or None

    def _is_allowed(self, character: str) -> bool:
        return character.isalnum() or character in self._allowed

    def _is_usable(self, line: str) -> bool:
        if not MIN_LINE_LENGTH <= len(line) <= MAX_LINE_LENGTH:
            return False
        if sum(1 for character in line if character.isalpha()) < MIN_LETTER_COUNT:
            return False
        return self._detector.is_dominated_by(line, self._script, MIN_SCRIPT_SHARE)


class CorpusBuilder:
    def __init__(self, extractor_seed: int) -> None:
        self._extractor_seed = extractor_seed

    def build(self, spec: CorpusSpec) -> list[str]:
        extractor = LineExtractor(spec.script, ScriptDetector(), np.random.default_rng(self._extractor_seed))
        collected: dict[str, None] = {}
        with Progress() as progress:
            task = progress.add_task(f"extracting {spec.name}", total=spec.line_target)
            for article in ParquetArticleStream(spec.url).iterate():
                for line in extractor.extract(article):
                    if line in collected:
                        continue
                    collected[line] = None
                    progress.advance(task)
                if len(collected) >= spec.line_target:
                    break
        return list(collected)[: spec.line_target]


def write_corpus(lines: list[str], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(lines) + "\n"
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build text line corpora from Wikipedia parquet shards")
    parser.add_argument("--output-dir", type=Path, default=Path("data/corpus"))
    parser.add_argument("--lock-path", type=Path, default=Path("data/corpus/corpus.lock.json"))
    parser.add_argument("--russian-lines", type=int, default=600_000)
    parser.add_argument("--english-lines", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    specs = [
        CorpusSpec("ru", "20231101.ru", 1, 21, Script.CYRILLIC, arguments.russian_lines),
        CorpusSpec("en", "20231101.en", 0, 41, Script.LATIN, arguments.english_lines),
    ]
    builder = CorpusBuilder(arguments.seed)
    records = []
    for spec in specs:
        lines = builder.build(spec)
        path = arguments.output_dir / f"{spec.name}.txt"
        digest = write_corpus(lines, path)
        records.append({
            "name": spec.name,
            "url": spec.url,
            "path": path.as_posix(),
            "line_count": len(lines),
            "sha256": digest,
        })
        print(f"{spec.name}: {len(lines)} lines -> {path}")
    arguments.lock_path.parent.mkdir(parents=True, exist_ok=True)
    arguments.lock_path.write_text(json.dumps({"corpora": records}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
