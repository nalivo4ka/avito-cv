from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from rich.progress import Progress

from avitocv.data.synthesis.fonts import FontRegistry, Script

METADATA_URL = "https://fonts.google.com/metadata/fonts"
CSS_URL_TEMPLATE = "https://fonts.googleapis.com/css2?family={spec}"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
REQUEST_TIMEOUT_SECONDS = 30
NETWORK_ERRORS = (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ConnectionError)
FACE_PATTERN = re.compile(r"@font-face\s*\{(.*?)\}", re.DOTALL)
STYLE_PATTERN = re.compile(r"font-style:\s*(\w+)")
WEIGHT_PATTERN = re.compile(r"font-weight:\s*(\d+)")
SOURCE_PATTERN = re.compile(r"url\((https://[^)]+\.ttf)\)")
ITALIC_SUFFIX = "i"
PREFERRED_STYLES = ("400", "700", "400i", "700i", "300", "500", "900")


@dataclass(frozen=True)
class FamilyMetadata:
    name: str
    styles: tuple[str, ...]
    popularity: int
    has_cyrillic: bool
    category: str

    @property
    def css_name(self) -> str:
        return self.name.replace(" ", "+")

    @property
    def file_stem(self) -> str:
        return self.name.replace(" ", "")


@dataclass(frozen=True)
class DownloadedFont:
    family: str
    style: str
    url: str
    path: Path
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict:
        return {
            "family": self.family,
            "style": self.style,
            "url": self.url,
            "path": self.path.as_posix(),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


class GoogleFontsCatalog:
    def fetch(self) -> list[FamilyMetadata]:
        payload = json.loads(self._strip_guard(self._read(METADATA_URL)))
        return [self._to_metadata(item) for item in payload["familyMetadataList"]]

    def _read(self, url: str) -> str:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            return response.read().decode("utf-8")

    def _strip_guard(self, raw: str) -> str:
        return raw[raw.index("{"):]

    def _to_metadata(self, item: dict) -> FamilyMetadata:
        return FamilyMetadata(
            name=item["family"],
            styles=tuple(item.get("fonts", {}).keys()),
            popularity=int(item.get("popularity", 10**6)),
            has_cyrillic="cyrillic" in item.get("subsets", []),
            category=str(item.get("category", "")),
        )


class FamilySelector:
    """Отбирает семейства: всю кириллицу и латиницу с упором на рисованные шрифты.

    Латиница раньше отбиралась только по популярности, то есть почти одни гротески. На тесте
    это вылезло: кропы с рисованными и декоративными надписями оказались среди самых трудных —
    у них ломаются типографские закономерности, на которые опирается модель, вроде более
    мелкой верхней чаши или приподнятой перекладины. Кириллических декоративных шрифтов в
    Google Fonts всего 63, и они уже все взяты; латинских — почти пятьсот, и они добавляют
    именно разнообразие форм, а не ещё один способ отрисовать то же самое.
    """

    DECORATIVE_CATEGORIES = ("Display", "Handwriting")

    def __init__(self, cyrillic_limit: int, latin_limit: int, decorative_limit: int = 0) -> None:
        self._cyrillic_limit = cyrillic_limit
        self._latin_limit = latin_limit
        self._decorative_limit = decorative_limit

    def select(self, families: list[FamilyMetadata]) -> list[FamilyMetadata]:
        chosen = self._take(families, True, self._cyrillic_limit) + self._take(families, False, self._latin_limit)
        taken = {family.name for family in chosen}
        decorative = [
            family for family in sorted(families, key=lambda item: item.popularity)
            if family.category in self.DECORATIVE_CATEGORIES and family.name not in taken
        ]
        limit = self._decorative_limit
        return chosen + (decorative if limit <= 0 else decorative[:limit])

    def _take(self, families: list[FamilyMetadata], has_cyrillic: bool, limit: int) -> list[FamilyMetadata]:
        matching = sorted(
            (family for family in families if family.has_cyrillic == has_cyrillic),
            key=lambda family: family.popularity,
        )
        return matching if limit <= 0 else matching[:limit]


class StyleSelector:
    def __init__(self, variants_per_family: int) -> None:
        self._variants_per_family = variants_per_family

    def select(self, family: FamilyMetadata) -> list[str]:
        available = [style for style in PREFERRED_STYLES if style in family.styles]
        chosen = available or list(family.styles)[:1]
        return chosen[: self._variants_per_family]


class GoogleFontsDownloader:
    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir

    def download_family(self, family: FamilyMetadata, styles: list[str]) -> list[DownloadedFont]:
        if not styles:
            return []
        css = self._fetch_css(family, styles)
        if css is None:
            return []
        stored = (self._store(family, face) for face in self._parse_faces(css))
        return [font for font in stored if font is not None]

    def _fetch_css(self, family: FamilyMetadata, styles: list[str]) -> str | None:
        axes = ";".join(sorted(self._to_axis(style) for style in styles))
        url = CSS_URL_TEMPLATE.format(spec=f"{family.css_name}:ital,wght@{axes}")
        return self._read_text(url)

    def _to_axis(self, style: str) -> str:
        is_italic = style.endswith(ITALIC_SUFFIX)
        weight = style[:-1] if is_italic else style
        return f"{int(is_italic)},{weight}"

    def _parse_faces(self, css: str) -> list[tuple[str, str]]:
        faces = []
        for block in FACE_PATTERN.findall(css):
            source = SOURCE_PATTERN.search(block)
            weight = WEIGHT_PATTERN.search(block)
            style = STYLE_PATTERN.search(block)
            if source is None or weight is None:
                continue
            suffix = ITALIC_SUFFIX if style is not None and style.group(1) == "italic" else ""
            faces.append((weight.group(1) + suffix, source.group(1)))
        return faces

    def _store(self, family: FamilyMetadata, face: tuple[str, str]) -> DownloadedFont | None:
        style, url = face
        path = self._output_dir / f"{family.file_stem}-{style}.ttf"
        data = path.read_bytes() if path.exists() else self._read_bytes(url)
        if data is None:
            return None
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return DownloadedFont(
            family=family.name,
            style=style,
            url=url,
            path=path,
            sha256=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data),
        )

    def _read_text(self, url: str) -> str | None:
        payload = self._read_bytes(url)
        return None if payload is None else payload.decode("utf-8")

    def _read_bytes(self, url: str) -> bytes | None:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                return response.read()
        except NETWORK_ERRORS:
            return None


class FontCollectionBuilder:
    def __init__(self, downloader: GoogleFontsDownloader, style_selector: StyleSelector, worker_count: int) -> None:
        self._downloader = downloader
        self._style_selector = style_selector
        self._worker_count = worker_count

    def build(self, families: list[FamilyMetadata]) -> list[DownloadedFont]:
        collected: list[DownloadedFont] = []
        with Progress() as progress, ThreadPoolExecutor(max_workers=self._worker_count) as pool:
            task = progress.add_task("downloading fonts", total=len(families))
            futures = [
                pool.submit(self._downloader.download_family, family, self._style_selector.select(family))
                for family in families
            ]
            for future in futures:
                collected.extend(future.result())
                progress.advance(task)
        return sorted(collected, key=lambda font: (font.family, font.style))


def write_lockfile(fonts: list[DownloadedFont], path: Path) -> None:
    payload = {"font_count": len(fonts), "fonts": [font.to_dict() for font in fonts]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Google Fonts and build a coverage-checked font index")
    parser.add_argument("--output-dir", type=Path, default=Path("data/fonts/google"))
    parser.add_argument("--index-path", type=Path, default=Path("data/fonts/index.json"))
    parser.add_argument("--lock-path", type=Path, default=Path("data/fonts/fonts.lock.json"))
    parser.add_argument("--cyrillic-families", type=int, default=0)
    parser.add_argument("--latin-families", type=int, default=80)
    parser.add_argument("--decorative-families", type=int, default=0,
                        help="рисованные и декоративные семейства сверх основного отбора, 0 — все")
    parser.add_argument("--variants-per-family", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    families = GoogleFontsCatalog().fetch()
    selected = FamilySelector(
        arguments.cyrillic_families, arguments.latin_families, arguments.decorative_families
    ).select(families)
    print(f"selected {len(selected)} families out of {len(families)}")
    builder = FontCollectionBuilder(
        GoogleFontsDownloader(arguments.output_dir),
        StyleSelector(arguments.variants_per_family),
        arguments.workers,
    )
    fonts = builder.build(selected)
    write_lockfile(fonts, arguments.lock_path)
    registry = FontRegistry.from_directories([arguments.output_dir])
    registry.save_index(arguments.index_path)
    print(f"downloaded {len(fonts)} files, indexed {len(registry)} usable")
    print(f"cyrillic {registry.count_for(Script.CYRILLIC)} latin {registry.count_for(Script.LATIN)}")


if __name__ == "__main__":
    main()
