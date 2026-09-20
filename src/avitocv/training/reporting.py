"""Живое отображение хода обучения в терминале.

Отдельный модуль, а не часть цикла: обучение одинаково запускается и фоном, и руками, и только
во втором случае нужен прогресс-бар. Цвета метрик выбраны под задачу — `score` это `1 - Brier`,
и уровень 0.75 соответствует константному предсказанию 0.5, то есть полному отсутствию знания.
Всё, что ниже, красное не для красоты: это значит, что модель хуже, чем «не знаю».
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from avitocv.training.observers import RunDescription, TrainingObserver

console = Console()

CHANCE_LEVEL_SCORE = 0.75
BAR_WIDTH = 30


def style_for_score(value: float) -> str:
    if value >= 0.97:
        return "bold green"
    if value >= 0.93:
        return "green"
    if value >= 0.85:
        return "yellow"
    if value > CHANCE_LEVEL_SCORE:
        return "bright_red"
    return "bold red"


def style_for_loss(value: float) -> str:
    if value < 0.15:
        return "bold green"
    if value < 0.30:
        return "green"
    if value < 0.50:
        return "yellow"
    return "red"


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}с"
    minutes, remainder = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}м{remainder:02d}с"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}ч{minutes:02d}м"


class RichProgressObserver(TrainingObserver):
    """Прогресс-бар по эпохам и батчам, строка результата на эпоху и сводная таблица в конце."""

    def __init__(self) -> None:
        self._progress: Progress | None = None
        self._epoch_task = None
        self._batch_task = None
        self._total_epochs = 0
        self._tracked_slice = ""

    def on_run_start(self, description: RunDescription) -> None:
        console.print(self._header(description))
        self._total_epochs = description.total_epochs
        self._tracked_slice = description.tracked_slice
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(bar_width=BAR_WIDTH),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=False,
        )
        self._progress.start()
        self._epoch_task = self._progress.add_task(
            "[bold blue]эпохи", total=description.total_epochs, completed=description.start_epoch
        )
        self._batch_task = self._progress.add_task("[dim]батчи", total=description.steps_per_epoch)

    def on_epoch_start(self, epoch: int, total_epochs: int, steps: int) -> None:
        self._progress.update(self._epoch_task, description=f"[bold blue]эпоха {epoch + 1}/{total_epochs}")
        self._progress.reset(self._batch_task, total=steps)

    def on_batch(self, step: int, loss: float, learning_rate: float) -> None:
        self._progress.update(
            self._batch_task,
            completed=step,
            description=f"[dim]батчи  loss={loss:.4f}  lr={learning_rate:.2e}",
        )

    def on_evaluation_start(self) -> None:
        self._progress.update(self._batch_task, description="[dim]оценка по срезам валидации…")

    def on_epoch_end(self, record, is_best: bool) -> None:
        self._progress.update(self._epoch_task, completed=record.index + 1)
        self._progress.console.print(self._epoch_line(record, is_best))

    def on_interrupt(self, epoch: int) -> None:
        self._stop()
        console.print()
        console.print(
            Panel(
                f"Состояние сохранено после эпохи [bold]{epoch}[/bold].\n"
                "Запустите ту же команду ещё раз, чтобы продолжить с этого места.",
                title="[bold yellow]Прервано[/bold yellow]",
                expand=False,
            )
        )

    def on_run_end(self, history: list, best_score: float) -> None:
        self._stop()
        if not history:
            return
        console.print()
        console.print(Rule("[bold]Итог обучения[/bold]", style="magenta"))
        console.print(self._summary_table(history, self._tracked_slice))
        total = sum(record.seconds for record in history)
        console.print(f"[dim]всего {format_duration(total)} · эпох {len(history)} · лучший {best_score:.5f}[/dim]")
        console.print()

    def _stop(self) -> None:
        if self._progress is not None:
            self._progress.stop()
            self._progress = None

    @staticmethod
    def _header(description: RunDescription) -> Panel:
        lines = [
            f"[bold]{description.architecture}[/bold]  "
            f"[dim]{description.parameter_count:,} параметров[/dim]".replace(",", " "),
            f"[dim]устройство:[/dim] {description.device}",
            f"[dim]эпох:[/dim] {description.total_epochs}"
            + (f"  [yellow]продолжение с {description.start_epoch}[/yellow]" if description.is_resumed else ""),
            f"[dim]сэмплов за эпоху:[/dim] {description.samples_per_epoch:,}".replace(",", " "),
        ]
        return Panel("\n".join(lines), title="[bold magenta]Обучение[/bold magenta]", expand=False)

    def _epoch_line(self, record, is_best: bool) -> Text:
        line = Text()
        line.append("  ✓ " if not is_best else "  ★ ", style="bold green" if is_best else "green")
        line.append(f"эпоха {record.index + 1}/{self._total_epochs}", style="bold")
        line.append("   loss=")
        line.append(f"{record.loss:.4f}", style=style_for_loss(record.loss))
        for name, value in record.scores.items():
            line.append(f"  {name}=")
            line.append(f"{value:.4f}", style=style_for_score(value))
        line.append(f"  T={record.temperature:.2f}", style="dim")
        line.append(f"  [{format_duration(record.seconds)}]", style="dim")
        return line

    @staticmethod
    def _summary_table(history: list, tracked_slice: str) -> Table:
        slice_names = list(history[-1].scores)
        table = Table(show_header=True, header_style="bold magenta", show_lines=False, padding=(0, 1))
        table.add_column("эпоха", style="bold", justify="right")
        table.add_column("loss", justify="right")
        for name in slice_names:
            table.add_column(name, justify="right")
        table.add_column("T", justify="right", style="dim")
        table.add_column("время", justify="right", style="dim")

        # Подсвечиваем эпоху, по которой выбирался чекпоинт, а не первую попавшуюся метрику:
        # иначе таблица показывала бы лучшей одну эпоху, а в best.pt лежала бы другая.
        key = tracked_slice if tracked_slice in slice_names else slice_names[0]
        best_index = max(range(len(history)), key=lambda position: history[position].scores.get(key, 0.0))
        for position, record in enumerate(history):
            cells = [str(record.index + 1), Text(f"{record.loss:.4f}", style=style_for_loss(record.loss))]
            cells += [Text(f"{record.scores.get(name, 0.0):.4f}", style=style_for_score(record.scores.get(name, 0.0)))
                      for name in slice_names]
            cells += [f"{record.temperature:.2f}", format_duration(record.seconds)]
            table.add_row(*cells, style="on grey23" if position == best_index else None)
        return table
