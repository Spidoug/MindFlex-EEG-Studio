from __future__ import annotations

import csv
import time
import tkinter as tk
from tkinter import filedialog, ttk
from pathlib import Path
from typing import Callable

from .bci import CentroidModel, FeatureExtractor, validate_model
from .calibration import CALIBRATION_PROTOCOLS, CalibrationEngine, CalibrationPhase
from .controller import EEGController
from .lab import analyze_replay, run_experiment
from .mental_text import StableInterpreter
from .neural_visual import COMMAND_IDS, COMMAND_SYMBOLS, CommandTestPhase, CommandTestSession, CursorArena, TemporalEvidenceFilter
from .neuro_runtime import PROFILES, NeuroRuntime
from .rules import NeuroStep, RuntimeState, WorkflowRules
from .ui_components import Card, PageHeader, ResponsiveCardGrid, ResponsiveSplitPane, ScrollableFrame, StepBar, clear_children, section_title

class NeuroControlView(ttk.Frame):
    """Canonical composer for the complete Neuro Control lifecycle."""

    def __init__(
        self,
        master,
        controller: EEGController,
        rules: WorkflowRules,
        translator,
        runtime: NeuroRuntime,
        on_state_change: Callable[[], None],
    ) -> None:
        super().__init__(master, style="Page.TFrame", padding=18)
        self.controller = controller
        self.rules = rules
        self.tr = translator
        self.runtime = runtime
        self.on_state_change = on_state_change
        self.stage = NeuroStep.SETUP_TRAINING
        self.status_var = tk.StringVar(value="")
        self._live_job = None
        self._communication_running = False
        self._stream_state = None
        self._replay_path: Path | None = None
        self._interpreter = StableInterpreter(self.runtime.vocabulary)
        self._calibration_engine: CalibrationEngine | None = None
        self._calibration_job = None
        self._calibration_profile = ""
        self._calibration_purpose = ""
        self._calibration_start_sample_count = 0
        self.calibration_status_var = tk.StringVar(value=self._t("calibration.idle"))
        self.calibration_progress_var = tk.StringVar(value="")

        # Visual BCI state. The classifier remains transport/UI independent;
        # these helpers only organize blind-test evidence and cursor dynamics.
        self._command_test = CommandTestSession(seed=time.time_ns())
        self._cursor_arena = CursorArena(seed=time.time_ns())
        self._cursor_running = False
        self._cursor_last_tick = time.monotonic()
        self._last_cursor_prediction = None
        self._cursor_filter = TemporalEvidenceFilter(COMMAND_IDS, window=8)
        self._command_log: list[str] = []
        self._last_calibration_visual = None
        self.runtime.ensure_automatic_models(self.rules)
        self._build()

    def _t(self, key: str, **values) -> str:
        return self.tr.t(key, **values)

    def _profile_name(self, profile: str) -> str:
        return {
            "concentration": self._t("setup.concentration"),
            "cursor": self._t("setup.cursor"),
            "communication": self._t("setup.vocabulary"),
        }[profile]

    def _feature_vector(self, seconds: float = 2.0):
        snap = self.controller.snapshot()
        if not self.rules.feature_snapshot_usable(snap):
            return None
        samples = self.controller.raw_values(seconds, max_samples=max(FeatureExtractor.MIN_SAMPLES, int(512 * seconds * 1.20)))
        try:
            return FeatureExtractor.from_raw(samples, sample_rate=512.0)
        except (ValueError, ArithmeticError):
            return None

    def runtime_state(self) -> RuntimeState:
        snap = self.controller.snapshot()
        threshold = self.rules.policy.minimum_validation_accuracy
        return RuntimeState(
            connected=snap.connected,
            receiving=snap.receiving,
            model_ready=self.runtime.model_ready,
            model_validated=self.runtime.any_validated(threshold),
            live_ready=self.runtime.live_ready(threshold),
            communication_ready=self.runtime.communication_ready(threshold),
        )

    def _build(self) -> None:
        PageHeader(self, self._t("neuro.title"), self._t("neuro.subtitle")).pack(fill="x")
        self.step_host = ttk.Frame(self, style="Page.TFrame")
        self.step_host.pack(fill="x", pady=(12, 12))
        self._render_stepbar()
        self.stage_host = ttk.Frame(self, style="Page.TFrame")
        self.stage_host.pack(fill="both", expand=True)
        ttk.Label(self, textvariable=self.status_var, style="InlineStatus.TLabel").pack(fill="x", pady=(12, 0))
        self._render_stage()

    def _render_stepbar(self) -> None:
        clear_children(self.step_host)
        steps = [
            (self._t("neuro.setup"), self._t("neuro.setup.short")),
            (self._t("neuro.validation"), self._t("neuro.validation.short")),
            (self._t("neuro.live"), self._t("neuro.live.short")),
            (self._t("neuro.communication"), self._t("neuro.communication.short")),
            (self._t("neuro.lab"), self._t("neuro.lab.short")),
        ]
        StepBar(
            self.step_host,
            steps,
            int(self.stage),
            lambda idx: self.rules.neuro_allowed(NeuroStep(idx), self.runtime_state()),
            self.navigate_neuro,
            show_subtitles=False,
            min_step_width=185,
        ).pack(fill="x")

    def navigate_neuro(self, target: int | NeuroStep) -> None:
        if self._calibration_engine is not None and self._calibration_engine.running:
            self.status_var.set(self._t("calibration.navigation_blocked"))
            return
        target = NeuroStep(int(target))
        if not self.rules.neuro_allowed(target, self.runtime_state()):
            self.status_var.set(self._t("neuro.blocked"))
            return
        self.stage = target
        self._render_stepbar()
        self._render_stage()

    def _render_stage(self) -> None:
        clear_children(self.stage_host)
        self._cancel_live_job()
        {
            NeuroStep.SETUP_TRAINING: self._build_setup_training,
            NeuroStep.VALIDATION: self._build_validation,
            NeuroStep.LIVE_CONTROL: self._build_live_control,
            NeuroStep.COMMUNICATION: self._build_communication,
            NeuroStep.LABORATORY: self._build_laboratory,
        }[self.stage]()

    def _cancel_live_job(self) -> None:
        if self._live_job is not None:
            try:
                self.after_cancel(self._live_job)
            except Exception:
                pass
            self._live_job = None
        self._communication_running = False
        self._command_test.stop()
        self._cursor_running = False
        self._cursor_arena.stop()
        self._cursor_filter.reset()

    def _label_display(self, label: str | None) -> str:
        if not label:
            return "—"
        key = f"label.{label}"
        return self._t(key) if self.tr.has_key(key) else str(label)

    @staticmethod
    def _symbol_for_label(label: str | None) -> str:
        if label in COMMAND_SYMBOLS:
            return COMMAND_SYMBOLS[label]
        if label == "focused":
            return "◎"
        if label == "relaxed":
            return "○"
        if label:
            return "Ψ"
        return "?"

    def _draw_command_symbol(self, canvas: tk.Canvas, label: str | None, subtitle: str = "") -> None:
        if not canvas.winfo_exists():
            return
        canvas.delete("all")
        width = max(320, canvas.winfo_width())
        height = max(180, canvas.winfo_height())
        canvas.create_text(
            width / 2,
            height * 0.40,
            text=self._symbol_for_label(label),
            font=("Segoe UI Symbol", 82, "bold"),
            fill="#172033",
        )
        canvas.create_text(
            width / 2,
            height * 0.73,
            text=self._label_display(label),
            font=("Segoe UI", 18, "bold"),
            fill="#263348",
        )
        if subtitle:
            canvas.create_text(
                width / 2,
                height * 0.90,
                text=subtitle,
                font=("Segoe UI", 10),
                fill="#667085",
            )

    def _draw_command_strip(self, canvas: tk.Canvas, prediction=None) -> None:
        if not canvas.winfo_exists():
            return
        canvas.delete("all")
        width = max(560, canvas.winfo_width())
        height = max(100, canvas.winfo_height())
        gap = 8.0
        cell = (width - gap * (len(COMMAND_IDS) + 1)) / len(COMMAND_IDS)
        scores = prediction.scores if prediction is not None else {}
        current = prediction.label if prediction is not None else None
        for index, label in enumerate(COMMAND_IDS):
            x0 = gap + index * (cell + gap)
            x1 = x0 + cell
            y0, y1 = 8.0, height - 8.0
            active = label == current
            canvas.create_rectangle(
                x0, y0, x1, y1,
                fill="#e9f2ff" if active else "#ffffff",
                outline="#2f6fed" if active else "#cfd7e6",
                width=3 if active else 1,
            )
            canvas.create_text(
                (x0 + x1) / 2, y0 + 29,
                text=COMMAND_SYMBOLS[label],
                font=("Segoe UI Symbol", 30, "bold"),
                fill="#1f4f99" if active else "#344054",
            )
            canvas.create_text(
                (x0 + x1) / 2, y0 + 61,
                text=self._label_display(label),
                font=("Segoe UI", 9, "bold"),
                fill="#344054",
            )
            value = max(0.0, min(1.0, float(scores.get(label, 0.0))))
            canvas.create_rectangle(x0 + 10, y1 - 12, x1 - 10, y1 - 7, fill="#eef2f7", outline="")
            canvas.create_rectangle(
                x0 + 10, y1 - 12, x0 + 10 + (x1 - x0 - 20) * value, y1 - 7,
                fill="#2f6fed", outline="",
            )

    def _draw_cursor_arena(self) -> None:
        if not hasattr(self, "cursor_canvas") or not self.cursor_canvas.winfo_exists():
            return
        canvas = self.cursor_canvas
        width = max(640, canvas.winfo_width())
        height = max(340, canvas.winfo_height())
        self._cursor_arena.set_bounds(width, height)
        canvas.delete("all")
        arena = self._cursor_arena
        canvas.create_line(width / 2, 0, width / 2, height, fill="#edf0f5")
        canvas.create_line(0, height / 2, width, height / 2, fill="#edf0f5")
        target_r = 28
        canvas.create_oval(
            arena.target_x - target_r, arena.target_y - target_r,
            arena.target_x + target_r, arena.target_y + target_r,
            outline="#2c7a4b", width=4,
        )
        canvas.create_text(
            arena.target_x, arena.target_y,
            text=self._t("neural.visual.target"),
            font=("Segoe UI", 9, "bold"), fill="#2c7a4b",
        )
        if len(arena.path) > 1:
            coords: list[float] = []
            for x, y in arena.path:
                coords.extend((x, y))
            canvas.create_line(*coords, fill="#b8c0cc", width=1)
        cursor_r = 11
        canvas.create_oval(
            arena.cursor_x - cursor_r, arena.cursor_y - cursor_r,
            arena.cursor_x + cursor_r, arena.cursor_y + cursor_r,
            fill="#285f9e", outline="#173a61", width=2,
        )

    def _format_test_confusion(self) -> str:
        matrix = self._command_test.confusion()
        ids = list(COMMAND_IDS)
        header = "      " + " ".join(f"{COMMAND_SYMBOLS[item]:>5}" for item in ids)
        lines = [header]
        for actual in ids:
            row = matrix.get(actual, {})
            values = " ".join(f"{int(row.get(predicted, 0)):>5}" for predicted in ids)
            lines.append(f"{COMMAND_SYMBOLS[actual]:>5} {values}")
        return "\n".join(lines)

    @staticmethod
    def _set_text_widget(widget: tk.Text, value: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", value)
        widget.configure(state="disabled")

    def _build_profile_selector(self, master, command) -> None:
        grid = ResponsiveCardGrid(master, min_card_width=210, max_columns=3, gap=8, style="Card.TFrame")
        grid.pack(fill="x")
        for profile in PROFILES:
            style = "StepCurrent.TButton" if profile == self.runtime.active_profile else "Step.TButton"
            grid.add(
                ttk.Button(
                    grid,
                    text=self._profile_name(profile),
                    style=style,
                    command=lambda p=profile: command(p),
                )
            )

    def _build_setup_training(self) -> None:
        scroll = ScrollableFrame(self.stage_host)
        scroll.pack(fill="both", expand=True)
        host = scroll.body
        section_title(host, self._t("setup.title"), self._t("setup.subtitle")).pack(fill="x", pady=(0, 10))

        selector = Card(host, padding=10)
        selector.pack(fill="x", pady=(0, 10))
        ttk.Label(selector, text=self._t("setup.module.select"), style="CardMetaStrong.TLabel").pack(anchor="w", pady=(0, 6))
        self._build_profile_selector(selector, self._select_setup_profile)

        workspace = ResponsiveSplitPane(
            host,
            primary_min_width=540,
            secondary_min_width=320,
            primary_weight=2,
            secondary_weight=1,
            gap=12,
        )
        workspace.pack(fill="x")

        visual = Card(workspace, padding=14, style="AccentCard.TFrame")
        ttk.Label(visual, text=self._t("calibration.title"), style="AccentCardTitle.TLabel").pack(anchor="w")
        ttk.Label(
            visual,
            textvariable=self.calibration_status_var,
            style="AccentCardBody.TLabel",
            wraplength=760,
        ).pack(anchor="w", pady=(4, 0))
        ttk.Label(
            visual,
            textvariable=self.calibration_progress_var,
            style="AccentCardBody.TLabel",
            wraplength=760,
        ).pack(anchor="w", pady=(2, 6))
        self.calibration_cue_canvas = tk.Canvas(
            visual,
            height=360,
            background="#ffffff",
            highlightthickness=1,
            highlightbackground="#d5dbe7",
        )
        self.calibration_cue_canvas.pack(fill="both", expand=True)
        self.calibration_cue_canvas.bind(
            "<Configure>",
            lambda _event: self._draw_command_symbol(
                self.calibration_cue_canvas,
                self._last_calibration_visual[0] if self._last_calibration_visual else None,
                self._last_calibration_visual[1] if self._last_calibration_visual else self._t("calibration.idle"),
            ),
        )
        self._draw_command_symbol(self.calibration_cue_canvas, None, self._t("calibration.idle"))

        controls = Card(workspace, padding=14)
        ttk.Label(controls, text=self._t("calibration.mode"), style="CardMetaStrong.TLabel").pack(anchor="w")
        self.calibration_mode_var = tk.StringVar(value=self._mode_display(self.runtime.calibration_mode))
        mode_combo = ttk.Combobox(
            controls,
            textvariable=self.calibration_mode_var,
            values=[self._mode_display(key) for key in CALIBRATION_PROTOCOLS],
            state="readonly",
        )
        mode_combo.pack(fill="x", pady=(5, 6))
        mode_combo.bind("<<ComboboxSelected>>", self._select_calibration_mode)
        ttk.Button(controls, text=self._t("calibration.stop"), command=self._cancel_calibration).pack(fill="x")
        ttk.Label(
            controls,
            text=self._t("calibration.help"),
            style="CardMeta.TLabel",
            wraplength=340,
            justify="left",
        ).pack(fill="x", anchor="w", pady=(8, 10))
        ttk.Separator(controls).pack(fill="x", pady=(0, 10))
        self.setup_module_host = ttk.Frame(controls, style="Card.TFrame")
        self.setup_module_host.pack(fill="x")
        workspace.set(visual, controls)
        self._render_setup_module()

    def _select_setup_profile(self, profile: str) -> None:
        if self._calibration_engine is not None and self._calibration_engine.running:
            self.status_var.set(self._t("calibration.navigation_blocked"))
            return
        if profile not in PROFILES:
            return
        self.runtime.active_profile = profile
        self._render_stage()

    def _profile_labels(self, profile: str) -> tuple[str, ...]:
        return self.runtime.labels_for(profile)

    def _render_setup_module(self) -> None:
        if not hasattr(self, "setup_module_host") or not self.setup_module_host.winfo_exists():
            return
        clear_children(self.setup_module_host)
        profile = self.runtime.active_profile if self.runtime.active_profile in PROFILES else "concentration"
        self.runtime.active_profile = profile
        labels = self._profile_labels(profile)

        ttk.Label(self.setup_module_host, text=self._profile_name(profile), style="CardTitle.TLabel").pack(anchor="w")
        help_key = {
            "concentration": "setup.concentration.help",
            "cursor": "setup.cursor.help",
            "communication": "setup.vocabulary.help",
        }[profile]
        ttk.Label(
            self.setup_module_host,
            text=self._t(help_key),
            style="CardMeta.TLabel",
            wraplength=340,
            justify="left",
        ).pack(anchor="w", fill="x", pady=(4, 8))

        ttk.Label(self.setup_module_host, text=self._automatic_model_summary(profile), style="CardMetaStrong.TLabel", wraplength=340).pack(anchor="w")
        if labels:
            ttk.Label(self.setup_module_host, text=self._sample_summary(profile, labels), style="CardMeta.TLabel", wraplength=340).pack(anchor="w", pady=(2, 8))

        if profile in {"concentration", "cursor"}:
            self._build_standard_training_module(self.setup_module_host, profile, labels)
        else:
            self._build_vocabulary_training_module(self.setup_module_host)

    def _build_standard_training_module(self, card, profile: str, labels: tuple[str, ...]) -> None:
        ttk.Separator(card).pack(fill="x", pady=(2, 8))
        ttk.Label(card, text=self._t("calibration.selected"), style="CardMetaStrong.TLabel").pack(anchor="w")
        selected = tk.StringVar(value=self._t(f"label.{labels[0]}"))
        combo = ttk.Combobox(card, textvariable=selected, values=[self._t(f"label.{label}") for label in labels], state="readonly")
        combo.current(0)
        combo.pack(fill="x", pady=(5, 6))
        ttk.Button(
            card,
            text=self._t("calibration.selected"),
            command=lambda c=combo, ids=labels, p=profile: self._start_training_calibration(p, (ids[c.current()],)),
        ).pack(fill="x")
        ttk.Button(
            card,
            text=self._t("calibration.full"),
            command=lambda ids=labels, p=profile: self._start_training_calibration(p, ids),
        ).pack(fill="x", pady=(6, 0))

    def _build_vocabulary_training_module(self, card) -> None:
        ttk.Separator(card).pack(fill="x", pady=(2, 8))
        self.vocab_label = tk.StringVar(value=self.runtime.selected_vocabulary_label)
        ttk.Label(card, text=self._t("setup.vocabulary"), style="CardMetaStrong.TLabel").pack(anchor="w")
        ttk.Entry(card, textvariable=self.vocab_label).pack(fill="x", pady=(4, 7))
        self.vocab_phrase = tk.StringVar(value=self.runtime.vocabulary.phrases.get(self.runtime.selected_vocabulary_label, ""))
        ttk.Label(card, text=self._t("communication.phrase"), style="CardMetaStrong.TLabel").pack(anchor="w")
        ttk.Entry(card, textvariable=self.vocab_phrase).pack(fill="x", pady=(4, 7))
        ttk.Button(card, text=self._t("action.add"), command=self._add_vocabulary).pack(fill="x")
        ttk.Button(card, text=self._t("calibration.selected"), command=self._capture_vocabulary).pack(fill="x", pady=(6, 0))
        ttk.Button(
            card,
            text=self._t("calibration.full"),
            command=lambda: self._start_training_calibration("communication", tuple(self.runtime.vocabulary.phrases)),
        ).pack(fill="x", pady=(6, 0))
        ttk.Label(card, text=self._vocab_summary(), style="CardMeta.TLabel").pack(anchor="w", pady=(7, 0))

    def _sample_summary(self, profile: str, labels) -> str:
        session = self.runtime.sessions[profile]
        if not labels:
            return self._t("setup.samples.none")
        return " · ".join(
            self._t(
                "calibration.summary.item",
                label=self._t(f"label.{label}") if self.tr.has_key(f"label.{label}") else label,
                trials=session.trial_count(label),
                epochs=session.count(label),
            )
            for label in labels
        )

    def _vocab_summary(self) -> str:
        return self._t("setup.vocabulary.count", count=len(self.runtime.vocabulary.phrases))

    def _automatic_model_summary(self, profile: str) -> str:
        session = self.runtime.sessions[profile]
        model = self.runtime.models.get(profile)
        if not model or not model.ready:
            return self._t("setup.model.none", count=session.total_trials)
        return self._t("setup.model.ready", labels=len(model.labels), count=session.total_trials)

    def _mode_display(self, key: str) -> str:
        return self._t(f"calibration.mode.{key}")

    def _select_calibration_mode(self, _event=None) -> None:
        display = self.calibration_mode_var.get()
        for key in CALIBRATION_PROTOCOLS:
            if self._mode_display(key) == display:
                self.runtime.calibration_mode = key
                break

    def _persist_profile_safely(self, profile: str) -> bool:
        try:
            self.runtime.persist_profile(profile)
            return True
        except (OSError, ValueError, TypeError, KeyError) as exc:
            self.status_var.set(self._t("session.error", error=str(exc)))
            return False

    def _start_training_calibration(self, profile: str, labels) -> None:
        self._start_calibration(profile, labels, purpose="training")

    def _start_calibration(self, profile: str, labels, *, purpose: str) -> None:
        if self._calibration_engine is not None and self._calibration_engine.running:
            self.status_var.set(self._t("calibration.already_running"))
            return
        clean_labels = tuple(dict.fromkeys(str(label).strip() for label in labels if str(label).strip()))
        if not clean_labels:
            self.status_var.set(self._t("calibration.no_labels"))
            return
        snap = self.controller.snapshot()
        if not self.rules.feature_snapshot_usable(snap):
            self.status_var.set(self._t("capture.blocked"))
            return
        protocol = CALIBRATION_PROTOCOLS[self.runtime.calibration_mode]
        if purpose == "validation":
            protocol = protocol.validation_copy()
            target = self.runtime.validation_sessions[profile]
            target.clear()
            self.runtime.validations.pop(profile, None)
            self.runtime.persist_profile(profile)
        else:
            target = self.runtime.sessions[profile]

        def feature_provider(epoch_seconds: float):
            vector = self._feature_vector(epoch_seconds)
            return None if vector is None else vector.tolist()

        def epoch_sink(label: str, vector: list[float], trial_id: str, _global_index: int, epoch_seconds: float) -> None:
            epoch_index = sum(1 for sample in target.samples if sample.trial_id == trial_id)
            target.add(
                label,
                vector,
                timestamp=time.time(),
                trial_id=trial_id,
                epoch_index=epoch_index,
                epoch_seconds=epoch_seconds,
            )
            if purpose == "validation":
                model = self.runtime.models.get(profile)
                if model and model.ready:
                    try:
                        prediction = model.predict(vector)
                    except Exception:
                        prediction = None
                    if prediction is not None:
                        self._update_validation_prediction(prediction)

        self._calibration_profile = profile
        self._calibration_purpose = purpose
        self._calibration_start_sample_count = len(target.samples)
        self._calibration_engine = CalibrationEngine(
            clean_labels,
            protocol,
            feature_provider,
            epoch_sink,
            seed=time.time_ns(),
        )
        self._calibration_engine.start()
        self._update_calibration_progress()
        self.status_var.set(self._t("calibration.started"))
        self._schedule_calibration_tick()

    def _schedule_calibration_tick(self) -> None:
        if self._calibration_job is not None:
            try:
                self.after_cancel(self._calibration_job)
            except Exception:
                pass
        self._calibration_job = self.after(100, self._calibration_tick)

    def _calibration_tick(self) -> None:
        self._calibration_job = None
        engine = self._calibration_engine
        if engine is None:
            return
        progress = engine.tick()
        self._update_calibration_progress(progress)
        if progress.phase == CalibrationPhase.COMPLETE:
            profile = self._calibration_profile
            purpose = self._calibration_purpose
            collected = progress.epochs_collected
            rejected = progress.epochs_rejected
            self._calibration_engine = None
            self._calibration_profile = ""
            self._calibration_purpose = ""
            self.calibration_status_var.set(self._t("calibration.complete"))
            self.calibration_progress_var.set(self._t("calibration.complete.detail", epochs=collected, rejected=rejected))
            self._last_calibration_visual = (None, self._t("calibration.complete"))
            if purpose == "validation":
                self._run_validation(profile)
            else:
                self._finalize_training(profile, collected=collected, rejected=rejected)
            return
        if progress.phase == CalibrationPhase.CANCELLED:
            self._calibration_engine = None
            self.status_var.set(self._t("calibration.cancelled"))
            return
        self._schedule_calibration_tick()

    def _update_calibration_progress(self, progress=None) -> None:
        engine = self._calibration_engine
        if engine is None:
            return
        progress = progress or engine.progress()
        label = progress.label
        display = self._t(f"label.{label}") if label and self.tr.has_key(f"label.{label}") else label
        phase_key = {
            CalibrationPhase.PREPARE: "calibration.phase.prepare",
            CalibrationPhase.TASK: "calibration.phase.task",
            CalibrationPhase.REST: "calibration.phase.rest",
        }.get(progress.phase, "calibration.idle")
        phase_text = self._t(phase_key, label=display)
        self.calibration_status_var.set(phase_text)
        self.calibration_progress_var.set(
            self._t(
                "calibration.progress",
                trial=progress.trial_number,
                total=progress.total_trials,
                seconds=progress.seconds_remaining,
                epochs=progress.epochs_collected,
                rejected=progress.epochs_rejected,
            )
        )
        self._last_calibration_visual = (label or None, phase_text)
        for canvas_name in ("calibration_cue_canvas", "validation_cue_canvas"):
            canvas = getattr(self, canvas_name, None)
            if canvas is not None and canvas.winfo_exists():
                self._draw_command_symbol(canvas, label or None, phase_text)

    def _cancel_calibration(self) -> None:
        if self._calibration_job is not None:
            try:
                self.after_cancel(self._calibration_job)
            except Exception:
                pass
            self._calibration_job = None
        if self._calibration_engine is not None:
            profile = self._calibration_profile
            purpose = self._calibration_purpose
            target = self.runtime.validation_sessions[profile] if purpose == "validation" and profile else (self.runtime.sessions[profile] if profile else None)
            changed = target is not None and len(target.samples) > self._calibration_start_sample_count
            self._calibration_engine.cancel()
            self._calibration_engine = None
            self._calibration_profile = ""
            self._calibration_purpose = ""
            self.calibration_status_var.set(self._t("calibration.cancelled"))
            self.calibration_progress_var.set("")
            self.status_var.set(self._t("calibration.cancelled"))
            self._last_calibration_visual = (None, self._t("calibration.cancelled"))
            if purpose == "training" and profile and changed:
                self._finalize_training(profile, collected=len(target.samples) - self._calibration_start_sample_count, rejected=0, refresh=False)
            elif purpose == "validation" and profile:
                self.runtime.validation_sessions[profile].clear()
                self.runtime.validations.pop(profile, None)
                self._persist_profile_safely(profile)
            if hasattr(self, "calibration_cue_canvas") and self.calibration_cue_canvas.winfo_exists():
                self._draw_command_symbol(self.calibration_cue_canvas, None, self._t("calibration.cancelled"))
            self._refresh_after_state_change()

    def _add_vocabulary(self) -> None:
        label = self.vocab_label.get().strip()
        phrase = self.vocab_phrase.get().strip()
        is_new_label = label not in self.runtime.vocabulary.phrases
        try:
            self.runtime.vocabulary.set_phrase(label, phrase)
        except ValueError:
            self.status_var.set(self._t("vocabulary.invalid"))
            return
        if is_new_label:
            self.runtime.invalidate_training("communication", delete_persisted=True)
        self.runtime.selected_vocabulary_label = label
        self._interpreter = StableInterpreter(self.runtime.vocabulary)
        self._persist_profile_safely("communication")
        self.status_var.set(self._t("vocabulary.added", phrase=phrase))
        self._refresh_after_state_change()

    def _capture_vocabulary(self) -> None:
        label = self.vocab_label.get().strip()
        if label not in self.runtime.vocabulary.phrases:
            self.status_var.set(self._t("vocabulary.add_first"))
            return
        self.runtime.selected_vocabulary_label = label
        self._start_training_calibration("communication", (label,))

    def _finalize_training(self, profile: str, *, collected: int = 0, rejected: int = 0, refresh: bool = True) -> None:
        labels = self._profile_labels(profile)
        session = self.runtime.sessions[profile]
        self.runtime.validations.pop(profile, None)
        self.runtime.validation_sessions[profile].clear()
        if self.rules.training_ready(session, labels):
            try:
                self.runtime.models[profile] = CentroidModel.train(session)
                self.status_var.set(self._t("setup.auto.trained", profile=self._profile_name(profile)))
            except Exception as exc:
                self.runtime.models.pop(profile, None)
                self.status_var.set(self._t("model.error", error=str(exc)))
        else:
            self.runtime.models.pop(profile, None)
            self.status_var.set(
                self._t(
                    "setup.auto.collecting",
                    profile=self._profile_name(profile),
                    trials=self.rules.policy.minimum_training_trials_per_class,
                    epochs=self.rules.policy.minimum_training_epochs_per_class,
                )
            )
        self._persist_profile_safely(profile)
        if refresh:
            self._refresh_after_state_change()

    def _update_validation_prediction(self, prediction) -> None:
        if not hasattr(self, "validation_prediction_var"):
            return
        label = self._label_display(prediction.label)
        self.validation_prediction_var.set(
            f"{self._t('neural.visual.prediction', label=label)} · {self._t('neural.visual.confidence', value=prediction.confidence * 100)}"
        )
        bar = getattr(self, "validation_confidence", None)
        if bar is not None and bar.winfo_exists():
            bar["value"] = prediction.confidence * 100

    def _refresh_after_state_change(self) -> None:
        self._render_stepbar()
        self._render_stage()
        self.on_state_change()

    def _build_validation(self) -> None:
        scroll = ScrollableFrame(self.stage_host)
        scroll.pack(fill="both", expand=True)
        host = scroll.body
        section_title(host, self._t("validation.title"), self._t("validation.subtitle")).pack(fill="x", pady=(0, 10))

        selector = Card(host, padding=10)
        selector.pack(fill="x", pady=(0, 10))
        ttk.Label(selector, text=self._t("validation.center"), style="CardMetaStrong.TLabel").pack(anchor="w", pady=(0, 6))
        self._build_profile_selector(selector, self._select_validation_profile)

        profile = self.runtime.active_profile if self.runtime.active_profile in PROFILES else "concentration"
        self.runtime.active_profile = profile
        model = self.runtime.models.get(profile)
        result = self.runtime.validations.get(profile)
        validation_session = self.runtime.validation_sessions[profile]
        threshold = self.rules.policy.minimum_validation_accuracy

        workspace = ResponsiveSplitPane(
            host,
            primary_min_width=560,
            secondary_min_width=300,
            primary_weight=2,
            secondary_weight=1,
            gap=12,
        )
        workspace.pack(fill="x")

        visual = Card(workspace, padding=14, style="AccentCard.TFrame")
        ttk.Label(visual, text=self._profile_name(profile), style="AccentCardTitle.TLabel").pack(anchor="w")
        ttk.Label(visual, textvariable=self.calibration_status_var, style="AccentCardBody.TLabel", wraplength=760).pack(anchor="w", pady=(4, 0))
        ttk.Label(visual, textvariable=self.calibration_progress_var, style="AccentCardBody.TLabel", wraplength=760).pack(anchor="w", pady=(2, 6))
        self.validation_cue_canvas = tk.Canvas(
            visual,
            height=430,
            background="#ffffff",
            highlightthickness=1,
            highlightbackground="#d5dbe7",
        )
        self.validation_cue_canvas.pack(fill="both", expand=True)
        self.validation_cue_canvas.bind(
            "<Configure>",
            lambda _event: self._draw_command_symbol(
                self.validation_cue_canvas,
                self._last_calibration_visual[0] if self._last_calibration_visual else None,
                self._last_calibration_visual[1] if self._last_calibration_visual else self._t("calibration.idle"),
            ),
        )
        self._draw_command_symbol(self.validation_cue_canvas, None, self._t("calibration.idle"))
        self.validation_prediction_var = tk.StringVar(value=self._t("neural.visual.no_prediction"))
        ttk.Label(visual, textvariable=self.validation_prediction_var, style="MetricLarge.TLabel", anchor="center").pack(fill="x", pady=(8, 3))
        self.validation_confidence = ttk.Progressbar(visual, maximum=100)
        self.validation_confidence.pack(fill="x")

        controls = Card(workspace, padding=14)
        ttk.Label(controls, text=self._t("validation.center.help"), style="CardMeta.TLabel", wraplength=330, justify="left").pack(fill="x", anchor="w", pady=(0, 8))
        ttk.Separator(controls).pack(fill="x", pady=(0, 8))
        ttk.Label(controls, text=self._t("validation.period_help"), style="CardMeta.TLabel", wraplength=330, justify="left").pack(fill="x", anchor="w", pady=(0, 10))
        if not model or not model.ready:
            ttk.Label(controls, text=self._t("validation.not_ready"), style="CardMetaStrong.TLabel", wraplength=330, justify="left").pack(fill="x", anchor="w", pady=(0, 10))
            ttk.Button(controls, text=self._t("validation.collect_run"), state="disabled").pack(fill="x")
        else:
            ttk.Label(controls, text=self._automatic_model_summary(profile), style="CardMetaStrong.TLabel", wraplength=330).pack(anchor="w", pady=(0, 6))
            ttk.Label(
                controls,
                text=self._t("validation.samples_trials", trials=validation_session.total_trials, epochs=len(validation_session.samples)),
                style="CardMeta.TLabel",
                wraplength=330,
            ).pack(anchor="w", pady=(0, 8))
            ttk.Button(
                controls,
                text=self._t("validation.collect_run"),
                command=lambda p=profile, ids=tuple(model.labels): self._start_calibration(p, ids, purpose="validation"),
            ).pack(fill="x")
            ttk.Button(controls, text=self._t("calibration.stop"), command=self._cancel_calibration).pack(fill="x", pady=(6, 0))
            if result:
                ttk.Separator(controls).pack(fill="x", pady=12)
                ttk.Label(
                    controls,
                    text=self._t("validation.result", accuracy=result.accuracy * 100, correct=result.correct, total=result.total),
                    style="MetricLarge.TLabel",
                    wraplength=330,
                ).pack(anchor="w")
                status_key = "validation.passed" if result.accuracy >= threshold else "validation.failed"
                ttk.Label(
                    controls,
                    text=self._t(status_key, threshold=threshold * 100),
                    style="CardMeta.TLabel",
                    wraplength=330,
                    justify="left",
                ).pack(fill="x", anchor="w", pady=(6, 0))
        workspace.set(visual, controls)

        bottom = ttk.Frame(host, style="Page.TFrame")
        bottom.pack(fill="x", pady=(10, 6))
        ttk.Button(bottom, text=self._t("action.previous"), command=lambda: self.navigate_neuro(NeuroStep.SETUP_TRAINING)).pack(side="left")
        ttk.Button(bottom, text=self._t("action.next"), command=lambda: self.navigate_neuro(NeuroStep.LIVE_CONTROL)).pack(side="right")

    def _select_validation_profile(self, profile: str) -> None:
        if self._calibration_engine is not None and self._calibration_engine.running:
            self.status_var.set(self._t("calibration.navigation_blocked"))
            return
        if profile in PROFILES:
            self.runtime.active_profile = profile
            self._render_stage()

    def _run_validation(self, profile: str) -> None:
        model = self.runtime.models.get(profile)
        if not model or not model.ready:
            self.status_var.set(self._t("model.need_train"))
            return
        session = self.runtime.validation_sessions[profile]
        minimum = self.rules.policy.minimum_validation_trials_per_class
        if not self.rules.validation_ready(session, model.labels):
            self.status_var.set(self._t("validation.not_enough_trials", count=minimum))
            self._refresh_after_state_change()
            return
        self.runtime.validations[profile] = validate_model(model, session)
        result = self.runtime.validations[profile]
        threshold = self.rules.policy.minimum_validation_accuracy
        self.status_var.set(
            self._t("validation.passed" if result.accuracy >= threshold else "validation.failed", threshold=threshold * 100)
        )
        self._persist_profile_safely(profile)
        self._refresh_after_state_change()

    def _build_live_control(self) -> None:
        section_title(self.stage_host, self._t("live.title"), self._t("live.subtitle")).pack(fill="x", pady=(0, 8))

        status = Card(self.stage_host, padding=10, style="AccentCard.TFrame")
        status.pack(fill="x", pady=(0, 8))
        ttk.Label(status, text=self._t("live.trigger"), style="AccentCardTitle.TLabel").pack(side="left")
        self.trigger_value = ttk.Label(status, text="—", style="AccentCardMetric.TLabel")
        self.trigger_value.pack(side="left", padx=(14, 8))
        self.trigger_conf = ttk.Label(status, text="", style="AccentCardBody.TLabel")
        self.trigger_conf.pack(side="left", fill="x", expand=True)

        self.live_notebook = ttk.Notebook(self.stage_host)
        self.live_notebook.pack(fill="both", expand=True)

        command_scroll = ScrollableFrame(self.live_notebook)
        test_scroll = ScrollableFrame(self.live_notebook)
        cursor_scroll = ScrollableFrame(self.live_notebook)
        self.live_notebook.add(command_scroll, text=self._t("neural.visual.commands"))
        self.live_notebook.add(test_scroll, text=self._t("neural.visual.test"))
        self.live_notebook.add(cursor_scroll, text=self._t("neural.visual.cursor"))
        command_page = command_scroll.body
        test_page = test_scroll.body
        cursor_page = cursor_scroll.body

        # Command panel: the recognized command is the dominant surface.
        self.command_strip_canvas = tk.Canvas(command_page, height=92, background="#f7f9fc", highlightthickness=0)
        self.command_strip_canvas.pack(fill="x", pady=(0, 8))
        self.command_strip_canvas.bind(
            "<Configure>",
            lambda _event: self._draw_command_strip(self.command_strip_canvas, self._last_cursor_prediction),
        )
        self._draw_command_strip(self.command_strip_canvas)

        command_split = ResponsiveSplitPane(
            command_page,
            primary_min_width=560,
            secondary_min_width=300,
            primary_weight=2,
            secondary_weight=1,
            gap=12,
        )
        command_split.pack(fill="x")
        command_card = Card(command_split, padding=12)
        ttk.Label(command_card, text=self._t("neural.visual.current"), style="CardTitle.TLabel").pack(anchor="w")
        self.command_symbol_canvas = tk.Canvas(
            command_card,
            height=400,
            background="#ffffff",
            highlightthickness=1,
            highlightbackground="#d5dbe7",
        )
        self.command_symbol_canvas.pack(fill="both", expand=True, pady=(7, 6))
        self.command_symbol_canvas.bind(
            "<Configure>",
            lambda _event: self._draw_command_symbol(
                self.command_symbol_canvas,
                self._last_cursor_prediction.label if self._last_cursor_prediction else None,
                self._t("neural.visual.no_prediction") if self._last_cursor_prediction is None else "",
            ),
        )
        self._draw_command_symbol(self.command_symbol_canvas, None, self._t("neural.visual.no_prediction"))
        self.command_value_label = ttk.Label(command_card, text=self._t("neural.visual.no_prediction"), style="MetricLarge.TLabel")
        self.command_value_label.pack(anchor="center")
        self.command_conf_label = ttk.Label(command_card, text="", style="CardMeta.TLabel")
        self.command_conf_label.pack(anchor="center", pady=(1, 0))

        evidence = Card(command_split, padding=12)
        ttk.Label(evidence, text=self._t("neural.visual.evidence"), style="CardTitle.TLabel").pack(anchor="w")
        self.command_evidence_bars = {}
        self.command_evidence_labels = {}
        for label in COMMAND_IDS:
            line = ttk.Frame(evidence, style="Card.TFrame")
            line.pack(fill="x", pady=(6, 0))
            ttk.Label(
                line,
                text=f"{COMMAND_SYMBOLS[label]} {self._label_display(label)}",
                style="CardMetaStrong.TLabel",
                width=13,
            ).pack(side="left")
            bar = ttk.Progressbar(line, maximum=100)
            bar.pack(side="left", fill="x", expand=True, padx=(5, 5))
            value = ttk.Label(line, text="", style="CardMeta.TLabel", width=5, anchor="e")
            value.pack(side="right")
            self.command_evidence_bars[label] = bar
            self.command_evidence_labels[label] = value
        ttk.Separator(evidence).pack(fill="x", pady=(12, 6))
        ttk.Label(evidence, text=self._t("neural.visual.log"), style="CardMetaStrong.TLabel").pack(anchor="w")
        self.command_log_list = tk.Listbox(evidence, height=5, borderwidth=0, highlightthickness=1)
        self.command_log_list.pack(fill="both", expand=True, pady=(5, 0))
        for item in self._command_log[-30:]:
            self.command_log_list.insert("end", item)
        command_split.set(command_card, evidence)

        # Blind figure test: cue/figure first, result details in the side panel.
        test_actions = Card(test_page, padding=9)
        test_actions.pack(fill="x", pady=(0, 8))
        self.test_trial_count_var = tk.IntVar(value=20)
        ttk.Label(test_actions, text=self._t("neural.visual.test.trials"), style="CardMetaStrong.TLabel").pack(side="left")
        ttk.Spinbox(test_actions, from_=5, to=100, increment=5, width=6, textvariable=self.test_trial_count_var).pack(side="left", padx=(5, 12))
        ttk.Button(test_actions, text=self._t("neural.visual.test.start"), command=self._start_command_test).pack(side="left")
        ttk.Button(test_actions, text=self._t("neural.visual.test.stop"), command=self._stop_command_test).pack(side="left", padx=(6, 0))

        test_split = ResponsiveSplitPane(
            test_page,
            primary_min_width=560,
            secondary_min_width=300,
            primary_weight=2,
            secondary_weight=1,
            gap=12,
        )
        test_split.pack(fill="x")
        test_left = Card(test_split, padding=12)
        self.test_canvas = tk.Canvas(
            test_left,
            height=420,
            background="#ffffff",
            highlightthickness=1,
            highlightbackground="#d5dbe7",
        )
        self.test_canvas.pack(fill="both", expand=True)
        self.test_canvas.bind("<Configure>", lambda _event: self._refresh_command_test_visual())
        self.test_state_label = ttk.Label(
            test_left,
            text=self._t("neural.visual.test.idle"),
            style="CardBody.TLabel",
            wraplength=760,
            justify="center",
        )
        self.test_state_label.pack(fill="x", pady=(6, 1))
        self.test_prediction_label = ttk.Label(test_left, text=self._t("neural.visual.no_prediction"), style="MetricLarge.TLabel")
        self.test_prediction_label.pack(pady=(1, 2))

        test_right = Card(test_split, padding=12)
        ttk.Label(test_right, text=self._t("neural.visual.test.result.title"), style="CardTitle.TLabel").pack(anchor="w")
        self.test_result_label = ttk.Label(
            test_right,
            text=self._t("neural.visual.test.idle"),
            style="CardMeta.TLabel",
            wraplength=330,
            justify="left",
        )
        self.test_result_label.pack(fill="x", pady=(7, 4))
        self.test_confidence = ttk.Progressbar(test_right, maximum=100)
        self.test_confidence.pack(fill="x")
        self.test_summary_label = ttk.Label(
            test_right,
            text=self._t("neural.visual.test.summary", correct=0, completed=0, accuracy=0),
            style="CardMetaStrong.TLabel",
            wraplength=330,
        )
        self.test_summary_label.pack(anchor="w", pady=(8, 6))
        ttk.Label(test_right, text=self._t("neural.visual.test.matrix"), style="CardMetaStrong.TLabel").pack(anchor="w")
        self.test_matrix = tk.Text(test_right, height=9, width=38, wrap="none", state="disabled", font=("Consolas", 9))
        self.test_matrix.pack(fill="both", expand=True, pady=(4, 0))
        self._set_text_widget(self.test_matrix, self._format_test_confusion())
        test_split.set(test_left, test_right)
        self._refresh_command_test_visual()

        # Mental cursor: arena receives the main workspace; actions and evidence
        # stay in a narrow side panel instead of consuming vertical canvas space.
        cursor_split = ResponsiveSplitPane(
            cursor_page,
            primary_min_width=600,
            secondary_min_width=280,
            primary_weight=3,
            secondary_weight=1,
            gap=12,
        )
        cursor_split.pack(fill="x")
        arena_card = Card(cursor_split, padding=12)
        arena_head = ttk.Frame(arena_card, style="Card.TFrame")
        arena_head.pack(fill="x", pady=(0, 6))
        self.cursor_prediction_label = ttk.Label(arena_head, text=self._t("neural.visual.no_prediction"), style="CardMetaStrong.TLabel")
        self.cursor_prediction_label.pack(side="left")
        self.cursor_hits_label = ttk.Label(arena_head, text=self._t("neural.visual.cursor.hits", hits=0), style="CardMeta.TLabel")
        self.cursor_hits_label.pack(side="right")
        self.cursor_canvas = tk.Canvas(
            arena_card,
            height=480,
            background="#ffffff",
            highlightthickness=1,
            highlightbackground="#d5dbe7",
        )
        self.cursor_canvas.pack(fill="both", expand=True)
        self.cursor_canvas.bind("<Configure>", lambda _event: self._draw_cursor_arena())

        cursor_controls = Card(cursor_split, padding=12)
        ttk.Label(cursor_controls, text=self._t("neural.visual.cursor"), style="CardTitle.TLabel").pack(anchor="w")
        ttk.Button(cursor_controls, text=self._t("neural.visual.cursor.start"), command=self._start_cursor).pack(fill="x", pady=(7, 0))
        ttk.Button(cursor_controls, text=self._t("neural.visual.cursor.stop"), command=self._stop_cursor).pack(fill="x", pady=(5, 0))
        ttk.Button(cursor_controls, text=self._t("neural.visual.cursor.new_target"), command=self._new_cursor_target).pack(fill="x", pady=(5, 0))
        ttk.Separator(cursor_controls).pack(fill="x", pady=10)
        ttk.Label(cursor_controls, text=self._t("neural.visual.cursor.threshold"), style="CardMetaStrong.TLabel").pack(anchor="w")
        self.cursor_confidence_var = tk.DoubleVar(value=0.25)
        ttk.Spinbox(
            cursor_controls,
            from_=0.10,
            to=0.90,
            increment=0.05,
            textvariable=self.cursor_confidence_var,
        ).pack(fill="x", pady=(4, 5))
        self.cursor_state_label = ttk.Label(
            cursor_controls,
            text=self._t("neural.visual.cursor.stopped"),
            style="CardMeta.TLabel",
            wraplength=280,
        )
        self.cursor_state_label.pack(fill="x", anchor="w")
        ttk.Separator(cursor_controls).pack(fill="x", pady=10)
        ttk.Label(cursor_controls, text=self._t("neural.visual.evidence"), style="CardMetaStrong.TLabel").pack(anchor="w")
        self.cursor_evidence_bars = {}
        self.cursor_evidence_labels = {}
        for label in COMMAND_IDS:
            cell = ttk.Frame(cursor_controls, style="Card.TFrame")
            cell.pack(fill="x", pady=(5, 0))
            value_label = ttk.Label(cell, text=f"{COMMAND_SYMBOLS[label]} 0%", width=6, style="CardMetaStrong.TLabel")
            value_label.pack(side="left")
            bar = ttk.Progressbar(cell, maximum=100)
            bar.pack(side="left", fill="x", expand=True, padx=(5, 0))
            self.cursor_evidence_labels[label] = value_label
            self.cursor_evidence_bars[label] = bar
        cursor_split.set(arena_card, cursor_controls)
        self._draw_cursor_arena()

        self._schedule_live()

    def _cursor_model_ready(self) -> bool:
        threshold = self.rules.policy.minimum_validation_accuracy
        return self.runtime.validated("cursor", threshold) and "cursor" in self.runtime.models

    def _start_command_test(self) -> None:
        if not self._cursor_model_ready():
            self.status_var.set(self._t("live.not_validated"))
            return
        if not self.controller.snapshot().receiving:
            self.status_var.set(self._t("stream.required"))
            return
        try:
            trials = int(self.test_trial_count_var.get())
        except (tk.TclError, ValueError, TypeError):
            trials = 20
        model = self.runtime.models["cursor"]
        labels = tuple(label for label in COMMAND_IDS if label in model.labels)
        if len(labels) < 2:
            self.status_var.set(self._t("live.not_validated"))
            return
        self._command_test = CommandTestSession(labels=labels, seed=time.time_ns())
        self._command_test.start(trials=max(5, trials))
        self.status_var.set(self._t("neural.visual.test.running"))
        self._refresh_command_test_visual()

    def _stop_command_test(self) -> None:
        self._command_test.stop()
        self.status_var.set(self._t("neural.visual.test.stopped"))
        self._refresh_command_test_visual()

    def _refresh_command_test_visual(self) -> None:
        if not hasattr(self, "test_canvas") or not self.test_canvas.winfo_exists():
            return
        snap = self._command_test.snapshot()
        cue = snap.cue or None
        if snap.phase == CommandTestPhase.PREPARE:
            subtitle = self._t("neural.visual.test.prepare", trial=snap.trial_number, total=snap.total_trials)
            self._draw_command_symbol(self.test_canvas, cue, subtitle)
            self.test_state_label.configure(text=subtitle)
        elif snap.phase == CommandTestPhase.CUE:
            subtitle = self._t(
                "neural.visual.test.cue",
                label=self._label_display(cue),
                seconds=snap.seconds_remaining,
            )
            self._draw_command_symbol(self.test_canvas, cue, subtitle)
            self.test_state_label.configure(text=subtitle)
        elif snap.phase == CommandTestPhase.REST:
            subtitle = self._t("neural.visual.test.rest", seconds=snap.seconds_remaining)
            self._draw_command_symbol(self.test_canvas, None, subtitle)
            self.test_state_label.configure(text=subtitle)
        elif snap.phase == CommandTestPhase.COMPLETE:
            self._draw_command_symbol(self.test_canvas, None, self._t("neural.visual.test.complete"))
            self.test_state_label.configure(text=self._t("neural.visual.test.complete"))
        else:
            self._draw_command_symbol(self.test_canvas, None, self._t("neural.visual.test.idle"))
            self.test_state_label.configure(text=self._t("neural.visual.test.idle"))

        result = snap.last_result
        if result is not None:
            predicted = self._label_display(result.predicted) if result.predicted else self._t("neural.visual.no_decision")
            verdict = self._t("neural.visual.correct") if result.correct else self._t("neural.visual.wrong")
            self.test_prediction_label.configure(text=self._t("neural.visual.prediction", label=predicted))
            self.test_result_label.configure(
                text=self._t(
                    "neural.visual.test.result",
                    verdict=verdict,
                    target=self._label_display(result.cue),
                    predicted=predicted,
                    confidence=result.confidence * 100,
                    observations=result.observations,
                )
            )
            self.test_confidence["value"] = result.confidence * 100
        else:
            self.test_prediction_label.configure(text=self._t("neural.visual.no_prediction"))
            self.test_result_label.configure(text=self._t("neural.visual.test.idle"))
            self.test_confidence["value"] = 0
        accuracy = (100.0 * snap.correct / snap.completed) if snap.completed else 0.0
        self.test_summary_label.configure(
            text=self._t("neural.visual.test.summary", correct=snap.correct, completed=snap.completed, accuracy=accuracy)
        )
        self._set_text_widget(self.test_matrix, self._format_test_confusion())

    def _start_cursor(self) -> None:
        if not self._cursor_model_ready():
            self.status_var.set(self._t("live.not_validated"))
            return
        if not self.controller.snapshot().receiving:
            self.status_var.set(self._t("stream.required"))
            return
        self._cursor_running = True
        self._cursor_arena.start()
        self._cursor_last_tick = time.monotonic()
        self.cursor_state_label.configure(text=self._t("neural.visual.cursor.active"))
        self.status_var.set(self._t("neural.visual.cursor.active"))
        self._draw_cursor_arena()

    def _new_cursor_target(self) -> None:
        self._cursor_arena.new_target()
        self._draw_cursor_arena()

    def _stop_cursor(self) -> None:
        self._cursor_running = False
        self._cursor_arena.stop()
        if hasattr(self, "cursor_state_label") and self.cursor_state_label.winfo_exists():
            self.cursor_state_label.configure(text=self._t("neural.visual.cursor.stopped"))
        self._draw_cursor_arena()

    def _append_command_log(self, prediction) -> None:
        if prediction is None:
            return
        if getattr(self, "_last_logged_command", None) == prediction.label:
            return
        self._last_logged_command = prediction.label
        text = f"{time.strftime('%H:%M:%S')}  {COMMAND_SYMBOLS.get(prediction.label, '•')}  {self._label_display(prediction.label)}  {prediction.confidence * 100:.0f}%"
        self._command_log.append(text)
        if len(self._command_log) > 100:
            del self._command_log[:-100]
        if hasattr(self, "command_log_list") and self.command_log_list.winfo_exists():
            self.command_log_list.insert("end", text)
            if self.command_log_list.size() > 30:
                self.command_log_list.delete(0, self.command_log_list.size() - 31)
            self.command_log_list.see("end")

    def _update_cursor_prediction_visuals(self, prediction) -> None:
        self._last_cursor_prediction = prediction
        if hasattr(self, "command_strip_canvas") and self.command_strip_canvas.winfo_exists():
            self._draw_command_strip(self.command_strip_canvas, prediction)
        if prediction is None:
            if hasattr(self, "command_symbol_canvas") and self.command_symbol_canvas.winfo_exists():
                self._draw_command_symbol(self.command_symbol_canvas, None, self._t("neural.visual.no_prediction"))
            if hasattr(self, "command_value_label"):
                self.command_value_label.configure(text=self._t("neural.visual.no_prediction"))
                self.command_conf_label.configure(text="")
            for label in COMMAND_IDS:
                if hasattr(self, "command_evidence_bars") and label in self.command_evidence_bars:
                    self.command_evidence_bars[label]["value"] = 0
                    self.command_evidence_labels[label].configure(text="")
                if hasattr(self, "cursor_evidence_bars") and label in self.cursor_evidence_bars:
                    self.cursor_evidence_bars[label]["value"] = 0
                    self.cursor_evidence_labels[label].configure(text=COMMAND_SYMBOLS[label])
            return
        display = self._label_display(prediction.label)
        if hasattr(self, "command_symbol_canvas") and self.command_symbol_canvas.winfo_exists():
            self._draw_command_symbol(
                self.command_symbol_canvas,
                prediction.label,
                self._t("neural.visual.confidence", value=prediction.confidence * 100),
            )
        if hasattr(self, "command_value_label"):
            self.command_value_label.configure(text=f"{COMMAND_SYMBOLS.get(prediction.label, '•')} {display}")
            self.command_conf_label.configure(text=self._t("neural.visual.confidence", value=prediction.confidence * 100))
        for label in COMMAND_IDS:
            value = max(0.0, min(1.0, float(prediction.scores.get(label, 0.0)))) * 100
            if hasattr(self, "command_evidence_bars") and label in self.command_evidence_bars:
                self.command_evidence_bars[label]["value"] = value
                self.command_evidence_labels[label].configure(text=f"{value:.0f}%")
            if hasattr(self, "cursor_evidence_bars") and label in self.cursor_evidence_bars:
                self.cursor_evidence_bars[label]["value"] = value
                self.cursor_evidence_labels[label].configure(text=f"{COMMAND_SYMBOLS[label]} {value:.0f}%")
        self._append_command_log(prediction)

    def _schedule_live(self) -> None:
        if not self.winfo_exists() or self.stage != NeuroStep.LIVE_CONTROL:
            return
        snap = self.controller.snapshot()
        threshold = self.rules.policy.minimum_validation_accuracy
        concentration_prediction = None
        cursor_prediction = None
        if self.rules.feature_snapshot_usable(snap):
            features = self._feature_vector(2.0)
            try:
                if features is None:
                    raise ValueError("RAW epoch is not ready")
                if self.runtime.validated("concentration", threshold):
                    concentration_prediction = self.runtime.models["concentration"].predict(features)
                    display = self._label_display(concentration_prediction.label)
                    self.trigger_value.configure(text=display)
                    self.trigger_conf.configure(text=self._t("live.confidence", value=concentration_prediction.confidence * 100))
                else:
                    self.trigger_value.configure(text="—")
                    self.trigger_conf.configure(text=self._t("live.not_validated"))
                if self.runtime.validated("cursor", threshold):
                    cursor_prediction = self.runtime.models["cursor"].predict(features)
            except (RuntimeError, ValueError, ArithmeticError) as exc:
                self.status_var.set(self._t("model.error", error=str(exc)))
        if not snap.receiving:
            self._cursor_filter.reset()
            stable_cursor_prediction = None
        else:
            stable_cursor_prediction = self._cursor_filter.push(cursor_prediction)
        self._update_cursor_prediction_visuals(stable_cursor_prediction)

        now = time.monotonic()
        if self._command_test.running:
            test_snap = self._command_test.tick(now)
            if test_snap.phase == CommandTestPhase.CUE:
                self._command_test.observe(cursor_prediction)
            self._refresh_command_test_visual()

        if self._cursor_running:
            dt = max(0.01, min(0.20, now - self._cursor_last_tick))
            self._cursor_last_tick = now
            try:
                confidence_floor = float(self.cursor_confidence_var.get())
            except (tk.TclError, ValueError, TypeError):
                confidence_floor = 0.25
            label = stable_cursor_prediction.label if stable_cursor_prediction is not None else None
            confidence = stable_cursor_prediction.confidence if stable_cursor_prediction is not None else 0.0
            self._cursor_arena.step(label, confidence, dt, threshold=confidence_floor)
            self.cursor_hits_label.configure(text=self._t("neural.visual.cursor.hits", hits=self._cursor_arena.hits))
            if stable_cursor_prediction is None:
                self.cursor_prediction_label.configure(text=self._t("neural.visual.no_prediction"))
            else:
                self.cursor_prediction_label.configure(
                    text=self._t(
                        "neural.visual.cursor.command",
                        symbol=COMMAND_SYMBOLS.get(stable_cursor_prediction.label, "•"),
                        label=self._label_display(stable_cursor_prediction.label),
                        confidence=stable_cursor_prediction.confidence * 100,
                    )
                )
            self._draw_cursor_arena()
        self._live_job = self.after(120, self._schedule_live)

    def _build_communication(self) -> None:
        scroll = ScrollableFrame(self.stage_host)
        scroll.pack(fill="both", expand=True)
        host = scroll.body
        section_title(host, self._t("communication.title"), self._t("communication.subtitle")).pack(fill="x", pady=(0, 12))
        card = Card(host)
        card.pack(fill="x")
        ttk.Label(card, text=self._t("communication.interpreter"), style="CardTitle.TLabel").pack(anchor="w")
        self.communication_output = tk.Text(card, height=14, wrap="word", state="disabled", borderwidth=0)
        self.communication_output.pack(fill="x", pady=(10, 10))
        actions = ResponsiveCardGrid(card, min_card_width=250, max_columns=2, gap=10, style="Card.TFrame")
        actions.pack(fill="x")
        primary = ttk.Frame(actions, style="Card.TFrame")
        self.communication_button = ttk.Button(primary, text=self._t("communication.start"), command=self._toggle_communication)
        self.communication_button.pack(side="left")
        ttk.Button(primary, text=self._t("communication.clear"), command=self._clear_communication_output).pack(side="left", padx=(6, 0))
        actions.add(primary)
        info = ttk.Frame(actions, style="Card.TFrame")
        ttk.Label(info, text=self._t("communication.subtitle"), style="CardMeta.TLabel", wraplength=520, justify="left").pack(fill="x", anchor="w")
        actions.add(info)

    def _toggle_communication(self) -> None:
        if self._communication_running:
            self._communication_running = False
            if self._live_job is not None:
                try:
                    self.after_cancel(self._live_job)
                except Exception:
                    pass
                self._live_job = None
            self.communication_button.configure(text=self._t("communication.start"))
            self.status_var.set(self._t("communication.stopped"))
            return
        threshold = self.rules.policy.minimum_validation_accuracy
        if not self.runtime.validated("communication", threshold):
            self.status_var.set(self._t("live.not_validated"))
            return
        if not self.controller.snapshot().receiving:
            self.status_var.set(self._t("stream.required"))
            return
        self._interpreter.reset()
        self._communication_running = True
        self.communication_button.configure(text=self._t("communication.stop"))
        self.status_var.set(self._t("communication.running"))
        self._communication_tick()

    def _communication_tick(self) -> None:
        if not self._communication_running or self.stage != NeuroStep.COMMUNICATION or not self.winfo_exists():
            return
        snap = self.controller.snapshot()
        if not snap.receiving:
            self._communication_running = False
            self.communication_button.configure(text=self._t("communication.start"))
            self.status_var.set(self._t("communication.stream_lost"))
            self._live_job = None
            return
        threshold = self.rules.policy.minimum_validation_accuracy
        if not self.runtime.validated("communication", threshold):
            self._communication_running = False
            self.communication_button.configure(text=self._t("communication.start"))
            self.status_var.set(self._t("live.not_validated"))
            self._live_job = None
            return
        if not self.rules.feature_snapshot_usable(snap):
            self.status_var.set(self._t("capture.blocked"))
            self._live_job = self.after(120, self._communication_tick)
            return
        try:
            features = self._feature_vector(2.0)
            if features is None:
                raise ValueError("RAW epoch is not ready")
            prediction = self.runtime.models["communication"].predict(features)
        except (RuntimeError, ValueError, ArithmeticError) as exc:
            self._communication_running = False
            self.communication_button.configure(text=self._t("communication.start"))
            self.status_var.set(self._t("model.error", error=str(exc)))
            self._live_job = None
            return
        phrase = self._interpreter.push(prediction)
        if phrase:
            self.communication_output.configure(state="normal")
            self.communication_output.insert("end", phrase + "\n")
            self.communication_output.see("end")
            self.communication_output.configure(state="disabled")
            self.status_var.set(self._t("communication.emitted", phrase=phrase))
        self._live_job = self.after(120, self._communication_tick)

    def _clear_communication_output(self) -> None:
        self.communication_output.configure(state="normal")
        self.communication_output.delete("1.0", "end")
        self.communication_output.configure(state="disabled")
        self._interpreter.reset()

    def _build_laboratory(self) -> None:
        scroll = ScrollableFrame(self.stage_host)
        scroll.pack(fill="both", expand=True)
        host = scroll.body
        section_title(host, self._t("lab.title"), self._t("lab.subtitle")).pack(fill="x", pady=(0, 12))
        grid = ResponsiveCardGrid(host, min_card_width=390, max_columns=2, gap=14)
        grid.pack(fill="x")
        replay = Card(grid)
        grid.add(replay)
        ttk.Label(replay, text=self._t("lab.replay"), style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(replay, text=self._t("lab.replay.help"), style="CardBody.TLabel", wraplength=620).pack(anchor="w", pady=(5, 10))
        ttk.Button(replay, text=self._t("lab.choose_file"), command=self._choose_replay).pack(anchor="w")
        self.replay_summary_var = tk.StringVar(value=self._t("lab.replay.none"))
        ttk.Label(replay, textvariable=self.replay_summary_var, style="CardMeta.TLabel", wraplength=620).pack(anchor="w", pady=(10, 0))

        experiments = Card(grid)
        grid.add(experiments)
        ttk.Label(experiments, text=self._t("lab.experiments"), style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(experiments, text=self._t("lab.experiments.help"), style="CardBody.TLabel", wraplength=620).pack(anchor="w", pady=(5, 10))
        ttk.Label(experiments, text=self._profile_name(self.runtime.active_profile), style="CardMetaStrong.TLabel").pack(anchor="w", pady=(0, 8))
        ttk.Button(experiments, text=self._t("lab.run_experiment"), command=self._run_experiment).pack(anchor="w")

    def _choose_replay(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), (self._t("file.all"), "*.*")])
        if not path:
            return
        self._replay_path = Path(path)
        try:
            summary = analyze_replay(self._replay_path)
        except (OSError, ValueError, csv.Error) as exc:
            self.replay_summary_var.set(self._t("lab.replay.error", error=str(exc)))
            self.status_var.set(self._t("lab.replay.error", error=str(exc)))
            return
        def metric(value):
            return "—" if value is None else f"{value:.1f}"
        self.replay_summary_var.set(
            self._t(
                "lab.replay.summary",
                frames=summary.frames,
                duration=summary.duration_seconds,
                attention=metric(summary.mean_attention),
                meditation=metric(summary.mean_meditation),
                bands=summary.active_bands,
            )
        )
        self.status_var.set(self._t("lab.replay.loaded", path=path))

    def _run_experiment(self) -> None:
        session = self.runtime.sessions[self.runtime.active_profile]
        try:
            result = run_experiment(session)
        except Exception as exc:
            self.status_var.set(self._t("model.error", error=str(exc)))
            return
        self.status_var.set(
            self._t(
                "lab.experiment.result",
                accuracy=result.validation.accuracy * 100,
                count=result.sample_count,
                train=result.train_count,
                validation=result.validation_count,
            )
        )

    def update_snapshot(self, snap) -> None:
        stream_state = (snap.connected, snap.receiving)
        if stream_state != self._stream_state:
            self._stream_state = stream_state
            self._render_stepbar()

    def on_show(self) -> None:
        self._render_stepbar()

    def on_hide(self) -> None:
        self._cancel_live_job()
        self._cancel_calibration()
