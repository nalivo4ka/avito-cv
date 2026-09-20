"""Письменности, которыми напечатан текст, и проверка принадлежности символов к ним.

Базовый словарь для двух других модулей: `fonts` решает по нему, покрывает ли шрифт нужную
письменность, а сборщик корпуса отбирает строки, написанные нужным алфавитом.
"""

from __future__ import annotations

from enum import Enum


class Script(Enum):
    """Письменность, которой напечатана строка."""

    CYRILLIC = "cyrillic"
    LATIN = "latin"


def _codepoint_range(first: str, last: str) -> tuple[int, ...]:
    return tuple(range(ord(first), ord(last) + 1))


# А..я лежат непрерывным блоком, а Ё и ё вынесены из него в другое место таблицы Unicode.
CYRILLIC_CODEPOINTS = _codepoint_range("\u0410", "\u044f") + (0x0401, 0x0451)
LATIN_CODEPOINTS = _codepoint_range("A", "Z") + _codepoint_range("a", "z")
DIGIT_CODEPOINTS = _codepoint_range("0", "9")

SCRIPT_CODEPOINTS: dict[Script, tuple[int, ...]] = {
    Script.CYRILLIC: CYRILLIC_CODEPOINTS,
    Script.LATIN: LATIN_CODEPOINTS,
}

_SCRIPT_CHARACTERS = {script: frozenset(chr(code) for code in codes) for script, codes in SCRIPT_CODEPOINTS.items()}


class ScriptDetector:
    """Считает долю букв заданной письменности в строке; работает фильтром корпуса."""

    def share_of(self, text: str, script: Script) -> float:
        letters = [character for character in text if character.isalpha()]
        if not letters:
            return 0.0
        alphabet = _SCRIPT_CHARACTERS[script]
        return sum(1 for character in letters if character in alphabet) / len(letters)

    def is_dominated_by(self, text: str, script: Script, threshold: float) -> bool:
        return self.share_of(text, script) >= threshold

    def dominant_script(self, text: str) -> Script | None:
        shares = {script: self.share_of(text, script) for script in Script}
        best = max(shares, key=shares.get)
        return best if shares[best] > 0.0 else None

