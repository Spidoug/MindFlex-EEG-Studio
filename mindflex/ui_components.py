from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable


class Card(ttk.Frame):
    def __init__(self, master, *, padding: int = 18, style: str = "Card.TFrame") -> None:
        super().__init__(master, padding=padding, style=style)


class PageHeader(ttk.Frame):
    def __init__(self, master, title: str, subtitle: str = "") -> None:
        super().__init__(master, style="Page.TFrame")
        ttk.Label(self, text=title, style="PageTitle.TLabel").pack(anchor="w")
        if subtitle:
            ttk.Label(self, text=subtitle, style="PageSubtitle.TLabel", wraplength=900).pack(anchor="w", pady=(4, 0))


class StepBar(ttk.Frame):
    """Responsive workflow step bar.

    Steps wrap into multiple rows instead of shrinking to unreadable widths.  The
    layout is recomputed from the actual content width, so translations with
    longer labels remain usable at the application's minimum window size.
    """

    def __init__(
        self,
        master,
        steps: list[tuple[str, str]],
        selected: int,
        allowed: Callable[[int], bool],
        command: Callable[[int], None],
        *,
        show_subtitles: bool = True,
        min_step_width: int = 220,
    ) -> None:
        super().__init__(master, style="Page.TFrame")
        self._steps = steps
        self._selected = selected
        self._allowed = allowed
        self._command = command
        self._show_subtitles = bool(show_subtitles)
        self._min_step_width = max(150, int(min_step_width))
        self._items: list[tuple[ttk.Frame, ttk.Button, ttk.Label | None]] = []
        for idx, (title, subtitle) in enumerate(steps):
            cell = ttk.Frame(self, style="Page.TFrame")
            style = "StepCurrent.TButton" if idx == selected else "Step.TButton"
            button = ttk.Button(
                cell,
                text=f"{idx + 1}. {title}",
                style=style,
                command=lambda i=idx: command(i),
                state="normal" if allowed(idx) else "disabled",
            )
            button.pack(fill="x")
            hint = None
            if self._show_subtitles and subtitle:
                hint = ttk.Label(cell, text=subtitle, style="StepHint.TLabel", wraplength=220, justify="left")
                hint.pack(fill="x", padx=8, pady=(4, 0))
            self._items.append((cell, button, hint))
        self.bind("<Configure>", self._layout)
        self.after_idle(self._layout)

    def _layout(self, _event=None) -> None:
        if not self.winfo_exists():
            return
        width = max(1, self.winfo_width())
        count = len(self._items)
        # Derive the layout from a single minimum readable width instead of
        # maintaining resolution-specific thresholds. Compact step bars can
        # therefore keep all workflow actions on one row whenever space allows.
        columns = max(1, min(count, width // self._min_step_width))
        for child, _, _ in self._items:
            child.grid_forget()
        for col in range(count):
            self.columnconfigure(col, weight=1 if col < columns else 0, uniform="steps" if col < columns else "")
        for idx, (cell, _, hint) in enumerate(self._items):
            row, col = divmod(idx, columns)
            if hint is not None:
                hint.configure(wraplength=max(150, int(width / columns) - 28))
            cell.grid(row=row, column=col, sticky="nsew", padx=(0 if col == 0 else 4, 4), pady=(0 if row == 0 else 8, 0))


class ResponsiveCardGrid(ttk.Frame):
    """A reusable grid that reflows cards based on available width."""

    def __init__(
        self,
        master,
        *,
        min_card_width: int = 340,
        max_columns: int = 3,
        gap: int = 14,
        style: str = "Page.TFrame",
    ) -> None:
        super().__init__(master, style=style)
        self.min_card_width = max(180, int(min_card_width))
        self.max_columns = max(1, int(max_columns))
        self.gap = max(0, int(gap))
        self._items: list[tk.Widget] = []
        self._columns = 0
        self.bind("<Configure>", self._layout)

    @property
    def columns(self) -> int:
        return self._columns or 1

    def add(self, widget: tk.Widget) -> tk.Widget:
        if widget not in self._items:
            self._items.append(widget)
        self.after_idle(self._layout)
        return widget

    def _layout(self, _event=None) -> None:
        if not self.winfo_exists():
            return
        width = max(1, self.winfo_width())
        columns = max(1, min(self.max_columns, (width + self.gap) // (self.min_card_width + self.gap)))
        # Avoid visually awkward layouts such as 3 + 1 when 2 + 2 is
        # available. A final single orphan is only kept for two-column grids.
        if columns > 2 and len(self._items) > columns and len(self._items) % columns == 1:
            columns -= 1
        self._columns = int(columns)
        for item in self._items:
            item.grid_forget()
        for col in range(self.max_columns):
            self.columnconfigure(col, weight=1 if col < columns else 0, uniform="responsive" if col < columns else "")
        rows = (len(self._items) + columns - 1) // columns
        for row in range(max(1, rows)):
            self.rowconfigure(row, weight=0)
        for idx, item in enumerate(self._items):
            row, col = divmod(idx, columns)
            left = 0 if col == 0 else self.gap // 2
            right = 0 if col == columns - 1 else self.gap // 2
            top = 0 if row == 0 else self.gap
            item.grid(row=row, column=col, sticky="nsew", padx=(left, right), pady=(top, 0))


class ResponsiveSplitPane(ttk.Frame):
    """Responsive visual-first split with an optional compact side panel.

    The primary surface receives more horizontal weight. When the requested
    minimum widths no longer fit, the secondary panel moves below the primary
    surface instead of squeezing either side into an unreadable column.
    """

    def __init__(
        self,
        master,
        *,
        primary_min_width: int = 520,
        secondary_min_width: int = 300,
        primary_weight: int = 2,
        secondary_weight: int = 1,
        gap: int = 14,
        style: str = "Page.TFrame",
    ) -> None:
        super().__init__(master, style=style)
        self.primary_min_width = max(260, int(primary_min_width))
        self.secondary_min_width = max(220, int(secondary_min_width))
        self.primary_weight = max(1, int(primary_weight))
        self.secondary_weight = max(1, int(secondary_weight))
        self.gap = max(0, int(gap))
        self.primary: tk.Widget | None = None
        self.secondary: tk.Widget | None = None
        self._horizontal = False
        self.bind("<Configure>", self._layout)

    @property
    def horizontal(self) -> bool:
        return self._horizontal

    def set(self, primary: tk.Widget, secondary: tk.Widget) -> None:
        self.primary = primary
        self.secondary = secondary
        self.after_idle(self._layout)

    def _layout(self, _event=None) -> None:
        if not self.winfo_exists() or self.primary is None or self.secondary is None:
            return
        width = max(1, self.winfo_width())
        horizontal = width >= self.primary_min_width + self.secondary_min_width + self.gap
        self._horizontal = horizontal
        self.primary.grid_forget()
        self.secondary.grid_forget()
        for col in range(2):
            self.columnconfigure(col, weight=0, uniform="")
        for row in range(2):
            self.rowconfigure(row, weight=0)
        if horizontal:
            self.columnconfigure(0, weight=self.primary_weight)
            self.columnconfigure(1, weight=self.secondary_weight)
            self.rowconfigure(0, weight=1)
            self.primary.grid(row=0, column=0, sticky="nsew", padx=(0, self.gap // 2))
            self.secondary.grid(row=0, column=1, sticky="nsew", padx=(self.gap // 2, 0))
        else:
            self.columnconfigure(0, weight=1)
            self.rowconfigure(0, weight=1)
            self.primary.grid(row=0, column=0, sticky="nsew")
            self.secondary.grid(row=1, column=0, sticky="nsew", pady=(self.gap, 0))


class StatusPill(ttk.Label):
    def set_state(self, text: str, state: str) -> None:
        style = {
            "ok": "StatusOk.TLabel",
            "warn": "StatusWarn.TLabel",
            "error": "StatusError.TLabel",
            "idle": "StatusIdle.TLabel",
        }.get(state, "StatusIdle.TLabel")
        self.configure(text=text, style=style)


def section_title(master, text: str, subtitle: str = "") -> ttk.Frame:
    frame = ttk.Frame(master, style="Page.TFrame")
    ttk.Label(frame, text=text, style="SectionTitle.TLabel").pack(anchor="w")
    if subtitle:
        ttk.Label(frame, text=subtitle, style="SectionSubtitle.TLabel", wraplength=850).pack(anchor="w", pady=(2, 0))
    return frame


def clear_children(widget) -> None:
    for child in widget.winfo_children():
        child.destroy()


class ScrollableFrame(ttk.Frame):
    """Responsive vertical scroll container for content that must never be compressed."""

    def __init__(self, master, *, style: str = "Page.TFrame") -> None:
        super().__init__(master, style=style)
        self.canvas = tk.Canvas(
            self, background="#f5f7fb", highlightthickness=0, borderwidth=0
        )
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.body = ttk.Frame(self.canvas, style=style)
        self._window = self.canvas.create_window((0, 0), window=self.body, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.body.bind("<Configure>", self._sync_scrollregion)
        self.canvas.bind("<Configure>", self._sync_width)
        self.canvas.bind("<Enter>", self._bind_wheel)
        self.canvas.bind("<Leave>", self._unbind_wheel)

    def _sync_scrollregion(self, _event=None) -> None:
        bbox = self.canvas.bbox("all")
        if bbox is not None:
            self.canvas.configure(scrollregion=bbox)

    def scroll_to_top(self) -> None:
        self.canvas.yview_moveto(0.0)

    def _sync_width(self, event) -> None:
        self.canvas.itemconfigure(self._window, width=max(1, event.width))

    def _bind_wheel(self, _event=None) -> None:
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_linux_wheel)
        self.canvas.bind_all("<Button-5>", self._on_linux_wheel)

    def _unbind_wheel(self, _event=None) -> None:
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _can_scroll(self) -> bool:
        first, last = self.canvas.yview()
        return first > 0.0 or last < 1.0

    def _on_mousewheel(self, event) -> None:
        if not self._can_scroll():
            return
        delta = -1 if event.delta > 0 else 1
        self.canvas.yview_scroll(delta * 3, "units")

    def _on_linux_wheel(self, event) -> None:
        if not self._can_scroll():
            return
        self.canvas.yview_scroll(-3 if event.num == 4 else 3, "units")
