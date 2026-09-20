"""
log_utils.py —— 基于 rich 的日志 / 终端展示工具箱
=================================================

本模块只关心「怎么把信息漂亮地讲出来」，不含任何训练逻辑，可被任意脚本复用。

快速上手
--------
    from log_utils import setup_logging, fmt_duration, fmt_pct

    log = setup_logging(name="MNIST", log_dir="logs")
    log.banner("MNIST 手写数字识别", "WideResNet-28 · 软投票集成")
    log.section("运行环境")
    log.kv("环境", [("Python", "3.12.14"), ("PyTorch", "2.11.0+cu128")])
    log.info("开始训练，共 [accent]5[/] 个模型")
    log.table("逐轮指标", ["Epoch", "Train", "Val"], [[1, "0.42", "99.02%"]])
    log.metrics([("集成准确率", "99.52%"), ("提升", "+0.31%")])
    log.success("全部完成")

    with log.progress() as progress:
        task = progress.add_task("训练", total=100, stats="")
        for i in range(100):
            progress.update(task, advance=1, stats="loss 0.12 · acc 98%")

设计要点
--------
* **双输出**：所有内容同时写到终端（带颜色、自适应宽度）与日志文件
  （纯文本、固定宽度、无 ANSI 转义），日志文件里顺序与终端完全一致。
* **级别可控**：``debug/info/warning/error/critical`` 受 ``LOG_LEVEL``（默认 INFO）控制；
  ``banner/section/table/kv`` 等展示组件始终输出，不受级别影响。
* **可复用**：环境变量 ``LOG_LEVEL``、构造参数 ``log_dir`` / ``log_file`` 决定落盘位置。
"""

from __future__ import annotations

import atexit
import logging
import math
import os
import re
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from rich import box
from rich.align import Align
from rich.cells import cell_len
from rich.columns import Columns
from rich.console import Console, Group, RenderableType
from rich.errors import MarkupError
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.prompt import Confirm, Prompt
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

__all__ = [
    "THEME",
    "LogManager",
    "setup_logging",
    "get_logger",
    "strip_markup",
    "fmt_int",
    "fmt_count",
    "fmt_params",
    "fmt_bytes",
    "fmt_duration",
    "fmt_eta",
    "fmt_pct",
    "fmt_lr",
    "fmt_signed",
    "heat_style",
    "model_summary",
]


# ─────────────────────────────────────────────────────────────
#  1. 主题：集中定义颜色，避免散落各处的魔法字符串
# ─────────────────────────────────────────────────────────────
THEME = Theme(
    {
        # 日志
        "log.time": "dim",
        "log.level.debug": "dim cyan",
        "log.level.info": "bold cyan",
        "log.level.warning": "bold yellow",
        "log.level.error": "bold red",
        "log.level.critical": "bold white on red",
        # 通用强调色
        "title": "bold magenta",
        "subtitle": "italic dim",
        "accent": "bold cyan",
        "head": "bold cyan",
        "ok": "bold green",
        "warn": "bold yellow",
        "bad": "bold red",
        "muted": "dim",
        "metric": "bold green",
        # 组件
        "kv.key": "dim",
        "kv.value": "bold",
        "panel.title": "bold cyan",
        "progress.description": "bold",
    },
    inherit=True,
)


# ─────────────────────────────────────────────────────────────
#  2. 格式化工具：把数字/时间变成人看的文本
# ─────────────────────────────────────────────────────────────
_MARKUP_RE = re.compile(r"\[/\]|\[/?[a-zA-Z#][^\[\]]*\]")


def strip_markup(text: str) -> str:
    """去掉 rich 标记（``[bold]`` / ``[/]`` 等），得到纯文本（写入日志文件时使用）。"""
    return _MARKUP_RE.sub("", text)


def fmt_int(n: float) -> str:
    """12345 → 12,345"""
    return f"{n:,.0f}"


def fmt_count(n: float, digits: int = 2) -> str:
    """59492 → 59.49K；1234567 → 1.23M"""
    for scale, suffix in ((1e9, "G"), (1e6, "M"), (1e3, "K")):
        if abs(n) >= scale:
            return f"{n / scale:.{digits}f}{suffix}"
    return f"{n:,.0f}"


def fmt_params(n: int) -> str:
    """参数量：1234567 → 1.23M"""
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.2f}K"
    return str(n)


def fmt_bytes(n: float | None, digits: int = 1) -> str:
    """字节数：25142624256 → 23.4 GiB"""
    if n is None:
        return "—"
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            return f"{value:.0f} B" if unit == "B" else f"{value:.{digits}f} {unit}"
        value /= 1024.0
    return f"{value:.{digits}f} TiB"  # pragma: no cover


def fmt_duration(seconds: float | None, digits: int = 1) -> str:
    """0.812 → 812ms；12.34 → 12.3s；185 → 3m05s；3725 → 1h02m05s"""
    if seconds is None or seconds < 0 or (isinstance(seconds, float) and math.isnan(seconds)):
        return "—"
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.{digits}f}s"
    total = int(round(seconds))
    minutes, sec = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{sec:02d}s"


def fmt_eta(seconds: float | None) -> str:
    """剩余时间（未知/已结束显示 —）。"""
    if seconds is None or seconds <= 0:
        return "—"
    return fmt_duration(seconds, digits=0)


def fmt_pct(x: float, digits: int = 2) -> str:
    """0.9902 → 99.02%"""
    return f"{x * 100:.{digits}f}%"


def fmt_lr(x: float) -> str:
    """0.001 → 1.00e-03"""
    return f"{x:.2e}"


def fmt_signed(x: float, digits: int = 4) -> str:
    """带符号输出：0.0123 → +0.0123"""
    return f"{x:+.{digits}f}"


def heat_style(value: float, vmax: float, *, invert: bool = False) -> str:
    """按数值大小返回由暗到亮的 rich 样式，用于混淆矩阵等热力着色。"""
    frac = 0.0 if vmax <= 0 else max(0.0, min(1.0, value / vmax))
    if invert:
        frac = 1.0 - frac
    frac = frac ** 0.6  # 略微提升低值区间的可辨识度
    r = int(38 + 95 * frac)
    g = int(58 + 150 * frac)
    b = int(78 + 120 * frac)
    fg = "black" if frac > 0.6 else "white"
    return f"{fg} on rgb({r},{g},{b})"


# ─────────────────────────────────────────────────────────────
#  3. logging.Handler：把标准 logging 记录渲染到双输出
# ─────────────────────────────────────────────────────────────
class _ConsoleHandler(logging.Handler):
    """将日志记录同时写到「终端 Console」与「纯文本日志文件 Console」。"""

    def __init__(self, manager: "LogManager") -> None:
        super().__init__()
        self.manager = manager

    # ---- 记录渲染 ----
    def render_record(self, record: logging.LogRecord, *, color: bool) -> Text:
        mgr = self.manager
        line = Text()
        if mgr.show_time:
            stamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
            line.append(f"[{stamp}] ", style="log.time" if color else "none")
        if mgr.show_level:
            style = f"log.level.{record.levelname.lower()}" if color else "none"
            line.append(f"[{record.levelname:<5}] ", style=style)

        message = record.getMessage()
        if color:
            try:
                line.append(Text.from_markup(message))
            except MarkupError:  # 消息里含有非 rich 语法的方括号
                line.append(message)
        else:
            line.append(strip_markup(message))
        return line

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D102
        try:
            mgr = self.manager
            mgr.console.print(self.render_record(record, color=True), markup=False, highlight=False)
            if mgr.file_console is not None:
                mgr.file_console.print(self.render_record(record, color=False), markup=False, highlight=False)
            if record.exc_info:
                if mgr.file_console is not None:
                    mgr.file_console.print(self.formatException(record.exc_info))
                mgr.console.print_exception(show_locals=False, width=mgr.console.width)
        except Exception:  # pragma: no cover - 日志失败不应影响主流程
            self.handleError(record)


# ─────────────────────────────────────────────────────────────
#  4. LogManager：统一的日志 + 展示门面
# ─────────────────────────────────────────────────────────────
class LogManager:
    """日志与终端展示的统一入口。

    Parameters
    ----------
    name:
        日志器名称，同时用于生成默认日志文件名（如 ``mnist_20260920_193000.log``）。
    level:
        日志级别，可被环境变量 ``LOG_LEVEL`` 覆盖。
    log_dir:
        日志目录；设为 ``None`` 时只输出到终端，不落盘。
    log_file:
        直接指定日志文件路径（优先级高于 ``log_dir``）。
    mirror_width:
        日志文件的渲染宽度（保持固定，避免表格被终端宽度影响）。
    """

    def __init__(
        self,
        name: str = "app",
        level: str | int = "INFO",
        log_dir: str | os.PathLike | None = "logs",
        log_file: str | os.PathLike | None = None,
        *,
        mirror_width: int = 110,
        show_time: bool = True,
        show_level: bool = True,
        capture_warnings: bool = True,
        console: Console | None = None,
    ) -> None:
        self.name = name
        self.show_time = show_time
        self.show_level = show_level
        self.started_at = time.perf_counter()
        self._closed = False

        self.console: Console = console or Console(theme=THEME, highlight=False, width=_console_width())

        # ---- 日志文件（纯文本镜像） ----
        self.log_path: Path | None = None
        self._file = None
        self.file_console: Console | None = None
        if log_dir is not None or log_file is not None:
            if log_file is not None:
                self.log_path = Path(log_file)
            else:
                directory = Path(log_dir)  # type: ignore[arg-type]
                directory.mkdir(parents=True, exist_ok=True)
                self.log_path = directory / f"{name.lower()}_{datetime.now():%Y%m%d_%H%M%S}.log"
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.log_path.open("w", encoding="utf-8")
            self.file_console = Console(
                theme=THEME,
                file=self._file,
                width=mirror_width,
                no_color=True,
                highlight=False,
                soft_wrap=False,
                legacy_windows=False,
            )
            atexit.register(self.close)

        # ---- 级别 ----
        raw_level = str(os.getenv("LOG_LEVEL") or level).upper()
        resolved = logging.getLevelName(raw_level)
        if not isinstance(resolved, int):
            resolved, raw_level = logging.INFO, "INFO"
        self.level: int = resolved
        self.level_name: str = raw_level

        # ---- 安装 handler（挂到 root，第三方库日志一并美化） ----
        self.handler = _ConsoleHandler(self)
        self.handler.setLevel(self.level)
        root = logging.getLogger()
        for existing in list(root.handlers):
            if isinstance(existing, _ConsoleHandler):
                root.removeHandler(existing)
        root.addHandler(self.handler)
        root.setLevel(self.level)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)
        self.logger.propagate = True

        if capture_warnings:
            logging.captureWarnings(True)

        if self.file_console is not None:
            self.file_console.print(
                f"# ===== 日志开始 {datetime.now():%Y-%m-%d %H:%M:%S} "
                f"| logger={name} | level={self.level_name} =====",
            )

    # ---------- 生命周期 ----------
    @property
    def debug_enabled(self) -> bool:
        """当前是否开启 DEBUG 级别（用于避免昂贵的调试字符串拼接）。"""
        return self.level <= logging.DEBUG

    @property
    def elapsed(self) -> float:
        """自 LogManager 创建以来的耗时（秒）。"""
        return time.perf_counter() - self.started_at

    def close(self) -> None:
        """写入收尾信息并关闭日志文件（幂等）。"""
        if self._closed:
            return
        self._closed = True
        if self.file_console is not None:
            self.file_console.print(
                f"# ===== 日志结束 {datetime.now():%Y-%m-%d %H:%M:%S} "
                f"| 总耗时 {fmt_duration(self.elapsed)} =====",
            )
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None

    # ---------- 日志级别接口 ----------
    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.critical(msg, *args, **kwargs)

    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """在 except 块中调用，自动附带 rich 彩色堆栈。"""
        kwargs.setdefault("exc_info", True)
        self.logger.error(msg, *args, **kwargs)

    def success(self, msg: str) -> None:
        """成功提示（带 ✔ 前缀）。"""
        self.logger.info(f"[ok]✔[/] {msg}")

    # ---------- 低层输出 ----------
    def _emit(self, renderable: RenderableType, **kwargs: Any) -> None:
        self.console.print(renderable, **kwargs)
        if self.file_console is not None:
            self.file_console.print(renderable, **kwargs)

    def render(self, renderable: RenderableType, **kwargs: Any) -> None:
        """直接输出任意 rich 可渲染对象（面板/表格/文本均可）。"""
        self._emit(renderable, **kwargs)

    def print(self, *objects: RenderableType, **kwargs: Any) -> None:
        self._emit(Group(*objects) if len(objects) > 1 else objects[0], **kwargs)

    def blank(self, count: int = 1) -> None:
        """输出空行（终端与日志文件同步）。"""
        for _ in range(count):
            self._emit("")

    # ---------- 标题 / 分区 ----------
    def rule(self, title: str | Text = "", *, char: str = "─", style: str = "muted", align: str = "left") -> None:
        renderable = Text.from_markup(title) if isinstance(title, str) else title
        self._emit(Rule(renderable, characters=char, style=style, align=align))  # type: ignore[arg-type]

    def banner(self, title: str, subtitle: str | None = None, *, char: str = "━") -> None:
        """页面顶部大标题。"""
        self.rule(char=char, style="accent")
        self._emit(Align.center(Text.from_markup(title)))
        if subtitle:
            self._emit(Align.center(Text.from_markup(subtitle, style="subtitle")))
        self.rule(char=char, style="accent")
        self.blank()

    def section(self, title: str, *, icon: str = "▌", style: str = "muted") -> None:
        """分区标题：▌ 数据集 ───────────────────"""
        head = Text()
        head.append(f"{icon} ", style="accent")
        head.append_text(Text.from_markup(title, style="head"))
        self._emit(Rule(head, characters="─", style=style, align="left"))

    # ---------- 组件 ----------
    def kv(
        self,
        title: str | None,
        items: Sequence[tuple[str, Any]] | Mapping[str, Any],
        *,
        columns: int = 1,
        border_style: str = "accent",
        footer: str | None = None,
        expand: bool = False,
        padding: int = 12,
    ) -> None:
        """键值面板：适合展示环境信息、超参数、运行摘要。"""
        pairs = list(items.items()) if isinstance(items, Mapping) else list(items)
        if not pairs:
            return
        # 用 cell_len 而非 len：中文/全角字符占 2 列，否则键名对不齐
        key_width = max(cell_len(strip_markup(str(k))) for k, _ in pairs)

        grid = Table.grid(padding=(0, 2))
        for _ in range(max(1, columns)):
            grid.add_column(justify="right", no_wrap=True, style="kv.key")
            grid.add_column(style="kv.value", overflow="fold")
        per_column = math.ceil(len(pairs) / max(1, columns))
        for row_index in range(per_column):
            row: list[Any] = []
            for column_index in range(max(1, columns)):
                index = column_index * per_column + row_index
                if index < len(pairs):
                    key, value = pairs[index]
                    label = strip_markup(str(key))
                    row.append(f"{' ' * (key_width - cell_len(label))}{label} :")
                    row.append(value)
                else:
                    row += ["", ""]
            grid.add_row(*row)

        panel = Panel(
            grid,
            title=Text.from_markup(f" {title} ", style="panel.title") if title else None,
            title_align="left",
            subtitle=Text.from_markup(f" {footer} ", style="muted") if footer else None,
            subtitle_align="right",
            border_style=border_style,
            box=box.ROUNDED,
            expand=expand,
            padding=(0, 1),
        )
        self._emit(panel)
        self.blank()

    def table(
        self,
        title: str | None,
        columns: Sequence[str | tuple[str, str]],
        rows: Sequence[Sequence[Any]],
        *,
        caption: str | None = None,
        box_style: Any = box.SIMPLE_HEAVY,
        header_style: str = "head",
        border_style: str = "muted",
        show_lines: bool = False,
        zebra: bool = False,
        expand: bool = False,
    ) -> None:
        """通用表格。

        ``columns`` 中的元素可以是 ``"名称"`` 或 ``("名称", "right")``（含对齐方式）。
        """
        renderable = Table(
            title=Text.from_markup(title) if isinstance(title, str) else title,
            box=box_style,
            header_style=header_style,
            border_style=border_style,
            caption=Text.from_markup(caption) if isinstance(caption, str) else caption,
            caption_justify="left",
            title_justify="left",
            title_style="panel.title",
            show_lines=show_lines,
            expand=expand,
            pad_edge=False,
            row_styles=["", "dim"] if zebra else None,
        )
        for spec in columns:
            if isinstance(spec, (tuple, list)):
                name, column_justify = spec[0], spec[1]
            else:
                name, column_justify = spec, "left"
            renderable.add_column(str(name), justify=column_justify, header_style=header_style)
        for row in rows:
            renderable.add_row(*[_as_cell(cell) for cell in row])
        self._emit(renderable)
        self.blank()

    def metrics(self, items: Sequence[tuple[str, Any]], *, border_style: str = "accent") -> None:
        """KPI 卡片：适合展示最终准确率、耗时等少量关键数字。"""
        cards = []
        for label, value in items:
            grid = Table.grid(expand=True)
            grid.add_column(justify="center")
            grid.add_row(Text.from_markup(str(value), style="metric", justify="center"))
            grid.add_row(Text.from_markup(str(label), style="muted", justify="center"))
            cards.append(Panel(grid, box=box.ROUNDED, border_style=border_style, padding=(0, 2)))
        self._emit(Columns(cards, equal=True, expand=True))
        self.blank()

    # ---------- 进度条 ----------
    def _progress_columns(self) -> tuple[Any, ...]:
        return (
            SpinnerColumn(style="accent"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None, complete_style="cyan", finished_style="green", pulse_style="accent"),
            TaskProgressColumn(),
            TextColumn("│ {task.fields[stats]}", style="muted", justify="left"),
            TextColumn("[muted]ETA[/]"),
            TimeRemainingColumn(compact=True),
        )

    @contextmanager
    def progress(self, *, columns: Sequence[Any] | None = None, transient: bool = False, disable: bool = False) -> Iterator[Progress]:
        """训练进度条上下文；任务的 ``stats`` 字段可写入任意富文本指标。"""
        progress = Progress(
            *(columns if columns is not None else self._progress_columns()),
            console=self.console,
            transient=transient,
            disable=disable,
            expand=False,
            refresh_per_second=8,
        )
        with progress:
            yield progress

    @staticmethod
    def add_task(
        progress: Progress,
        description: str,
        total: int | None = None,
        *,
        stats: str = "",
    ) -> TaskID:
        """创建进度任务（自动填充 ``stats`` 字段，避免模板 KeyError）。"""
        return progress.add_task(description, total=total, stats=stats)

    # ---------- 交互 ----------
    def ask_yes_no(self, prompt: str, *, default: bool = True) -> bool:
        """带默认值的 y/n 询问；无交互环境（EOF）时返回默认值。"""
        hint = "[y/n]" if default else "[y/N]"
        try:
            answer = Confirm.ask(f"[accent]{prompt}[/] {hint}", console=self.console, default=default)
            self.blank()
            return answer
        except (EOFError, KeyboardInterrupt):
            self.console.print()
            self.warning(f"未检测到交互输入，按默认值 {'继续' if default else '取消'}。")
            return default

    def ask_text(self, prompt: str, *, default: str = "") -> str:
        """文本输入（带默认值）。"""
        try:
            return Prompt.ask(f"[accent]{prompt}[/]", console=self.console, default=default)
        except (EOFError, KeyboardInterrupt):
            self.console.print()
            return default

    # ---------- 计时 ----------
    @contextmanager
    def timed(self, message: str, *, level: int = logging.INFO) -> Iterator[None]:
        """计时上下文：进入打点、结束汇报耗时，异常时也会给出耗时。"""
        start = time.perf_counter()
        self.logger.log(level, f"{message} …")
        try:
            yield
        except Exception as exc:  # noqa: BLE001 - 需要原样抛出
            self.logger.error(
                f"{message} [bad]失败[/] · {type(exc).__name__}: {exc} · "
                f"耗时 {fmt_duration(time.perf_counter() - start)}"
            )
            raise
        self.logger.log(level, f"{message} [ok]完成[/] · 耗时 [accent]{fmt_duration(time.perf_counter() - start)}[/]")


def _as_cell(cell: Any) -> Any:
    """把任意单元格内容转成 rich 可渲染的文本/对象。"""
    if isinstance(cell, (str, Text, Table)) or hasattr(cell, "__rich_console__"):
        return cell
    return str(cell)


def _console_width() -> int | None:
    """重定向到文件 / CI 时，允许用 COLUMNS 环境变量指定渲染宽度。"""
    raw = os.getenv("COLUMNS")
    if raw and raw.isdigit() and int(raw) > 20:
        return int(raw)
    return None


# ─────────────────────────────────────────────────────────────
#  5. 模型结构摘要（依赖 torch，按需导入）
# ─────────────────────────────────────────────────────────────
def model_summary(
    log: LogManager,
    model: Any,
    input_shape: tuple[int, ...] = (1, 1, 28, 28),
    *,
    device: Any = None,
    title: str = "模型结构",
) -> Table:
    """用 forward hook 统计各子模块的输入/输出形状与参数量，并打印表格。

    ``input_shape`` 为完整的输入张量形状（含 batch 维度），如 ``(1, 1, 28, 28)``。
    """
    import torch  # 延迟导入，保持 log_utils 在无 torch 环境可用

    if device is None:
        try:
            device = next(model.parameters()).device  # 与模型保持一致，避免设备不匹配
        except StopIteration:  # pragma: no cover - 无参数模型
            device = None

    record: dict[str, tuple[str, str]] = {}
    handles = []

    def describe(value: Any) -> str:
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        if not isinstance(value, torch.Tensor):
            return "—"
        if value.dim() == 0:
            return "scalar"
        if value.dim() == 1:
            return str(value.shape[0])
        return "×".join(str(s) for s in value.shape[1:])

    for child_name, child in model.named_children():

        def hook(module: Any, inputs: Any, output: Any, *, _name: str = child_name) -> None:
            record[_name] = (describe(inputs), describe(output))

        handles.append(child.register_forward_hook(hook))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            dummy = torch.zeros(input_shape, device=device) if device is not None else torch.zeros(input_shape)
            model(dummy)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rows = []
    for name, child in model.named_children():
        params = sum(p.numel() for p in child.parameters(recurse=True))
        shape_in, shape_out = record.get(name, ("—", "—"))
        share = f"{params / total * 100:.1f}%" if total else "—"
        rows.append([f"{name} [muted]({type(child).__name__})[/]", shape_in, shape_out, fmt_params(params), share])

    input_desc = "×".join(str(s) for s in input_shape)
    table = Table(
        title=Text.from_markup(f" {title} "),
        box=box.SIMPLE_HEAVY,
        header_style="head",
        border_style="muted",
        title_justify="left",
        title_style="panel.title",
        caption=Text.from_markup(
            f"输入 {input_desc} · 合计 {fmt_params(total)} 参数"
            f"（可训练 {fmt_params(trainable)}）"
        ),
        caption_justify="left",
        pad_edge=False,
    )
    header: Sequence[str | tuple[str, str]] = (
        "子模块",
        "输入形状",
        "输出形状",
        ("参数量", "right"),
        ("占比", "right"),
    )
    for spec in header:
        name, justify = (spec[0], spec[1]) if isinstance(spec, tuple) else (spec, "left")
        table.add_column(name, justify=justify, header_style="head")
    for row in rows:
        table.add_row(*[_as_cell(cell) for cell in row])
    log.render(table)
    log.blank()
    return table


# ─────────────────────────────────────────────────────────────
#  6. 模块级便捷入口
# ─────────────────────────────────────────────────────────────
_DEFAULT: LogManager | None = None


def setup_logging(**kwargs: Any) -> LogManager:
    """创建（并记住）全局 LogManager。"""
    global _DEFAULT
    _DEFAULT = LogManager(**kwargs)
    return _DEFAULT


def get_logger() -> LogManager:
    """获取全局 LogManager；若尚未创建则用默认参数创建。"""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = LogManager()
    return _DEFAULT
