from __future__ import annotations

import queue
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

from .bci import ModelStore
from .transport import EndpointInfo, TransportMode, discover_endpoints, endpoint_from_id, manual_endpoint
from .bluetooth_transport import BluetoothWorker
from .controller import EEGController, EEGSnapshot
from .diagnostics import DiagnosticSequence, DiagnosticStatus
from .i18n import LANGUAGES, Translator
from .lab import SessionRecorder
from .neuro_runtime import NeuroRuntime
from .neuro_ui import NeuroControlView
from .parser import EEG_BANDS
from .plotting import FastLinePlot, FastMultiLinePlot, FrameScheduler, configure_plot_font
from .rules import RuntimeState, WorkflowRules, WorkflowStep
from .settings import APP_AUTHOR, APP_HANDLE, APP_NAME, Settings
from .ui_components import Card, PageHeader, ResponsiveCardGrid, ScrollableFrame, StatusPill, clear_children
from .user_profile import UserProfile


class ConnectionView(ttk.Frame):
    """Connection page for live Bluetooth Classic and Serial/USB transports.

    Bluetooth discovery is continuous: DeviceWatcher maintains the live table,
    selection is explicit, and RFCOMM is validated before Monitor is enabled.
    """

    def __init__(self, master, app: "MindFlexApp") -> None:
        super().__init__(master, style="Page.TFrame", padding=24)
        self.app = app
        self.tr = app.tr
        mode = app.settings.connection_mode
        if mode not in {"bluetooth", "serial"}:
            mode = "bluetooth"
        self.mode_var = tk.StringVar(value=self._mode_label(mode))
        self.endpoint_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="")
        self._endpoint_by_display: dict[str, EndpointInfo] = {}

        # Live Bluetooth table state owned by the persistent worker.
        self._bt_events = app.bluetooth_events
        self._bt_worker = app.bluetooth_worker
        self._bt_device_rows: dict[str, str] = {}
        self._bt_row_devices: dict[str, str] = {}
        self._bt_row_counter = 0
        self._bt_scan_active = False
        self._connect_running = False
        self._hidden = False
        self._bt_poll_job = None

        self._build()
        self._apply_mode_ui(initial=True)
        self._bt_poll_job = self.after(60, self._poll_bluetooth_events)

    def t(self, key, **values):
        return self.tr.t(key, **values)

    def _mode_label(self, mode: str) -> str:
        return self.t("connection.mode.serial") if mode == "serial" else self.t("connection.mode.bluetooth")

    def _selected_mode(self) -> TransportMode:
        if self.mode_var.get() == self.t("connection.mode.serial"):
            return TransportMode.SERIAL
        return TransportMode.BLUETOOTH

    def _build(self) -> None:
        PageHeader(self, self.t("connection.title"), self.t("connection.subtitle")).pack(fill="x")
        scroll = ScrollableFrame(self)
        scroll.pack(fill="both", expand=True, pady=(18, 0))
        host = scroll.body
        self._page_scroll = scroll

        mode_card = Card(host)
        mode_card.pack(fill="x", pady=(0, 12))
        top = ttk.Frame(mode_card, style="Card.TFrame")
        top.pack(fill="x")
        ttk.Label(top, text=f"1. {self.t('connection.device')}", style="CardTitle.TLabel").pack(side="left")
        ttk.Button(top, text=self.t("connection.simulator"), command=self.start_simulator).pack(side="right")
        ttk.Label(
            mode_card, text=self.t("connection.device.help"), style="CardBody.TLabel", wraplength=980
        ).pack(anchor="w", pady=(5, 10))
        self.mode_combo = ttk.Combobox(
            mode_card,
            textvariable=self.mode_var,
            values=[self.t("connection.mode.bluetooth"), self.t("connection.mode.serial")],
            state="readonly",
            width=30,
        )
        self.mode_combo.pack(anchor="w")
        self.mode_combo.bind("<<ComboboxSelected>>", self._mode_changed)

        self.bt_card = Card(host)
        bt_header = ttk.Frame(self.bt_card, style="Card.TFrame")
        bt_header.pack(fill="x")
        title_box = ttk.Frame(bt_header, style="Card.TFrame")
        title_box.pack(side="left", fill="x", expand=True)
        ttk.Label(title_box, text=self.t("connection.bluetooth.title"), style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(
            title_box, text=self.t("connection.bluetooth.help"), style="CardBody.TLabel", wraplength=900
        ).pack(anchor="w", pady=(4, 0))
        self.bt_connection_pill = StatusPill(bt_header)
        self.bt_connection_pill.pack(side="right", padx=(12, 0))

        actions = ttk.Frame(self.bt_card, style="Card.TFrame")
        actions.pack(fill="x", pady=(12, 10))
        left_actions = ttk.Frame(actions, style="Card.TFrame")
        left_actions.pack(side="left", fill="x", expand=True)
        self.bt_search_button = ttk.Button(left_actions, text=self.t("action.search"), command=self.search_bluetooth)
        self.bt_search_button.pack(side="left")
        self.bt_connect_button = ttk.Button(
            left_actions, text=self.t("action.connect"), command=self.connect_bluetooth_selected, style="Primary.TButton"
        )
        self.bt_connect_button.pack(side="left", padx=6)
        self.bt_disconnect_button = ttk.Button(left_actions, text=self.t("action.disconnect"), command=self.disconnect)
        self.bt_disconnect_button.pack(side="left")
        ttk.Label(actions, textvariable=self.status_var, style="CardMeta.TLabel", wraplength=360).pack(side="right", padx=(12, 0))

        table = ttk.Frame(self.bt_card, style="Card.TFrame")
        table.pack(fill="both")
        table.columnconfigure(0, weight=1)
        table.rowconfigure(0, weight=1)
        columns = ("name", "address", "paired", "present", "connected", "signal")
        self.bt_tree = ttk.Treeview(table, columns=columns, show="headings", selectmode="browse", height=12)
        headings = {
            "name": self.t("connection.bluetooth.column.name"),
            "address": self.t("connection.bluetooth.column.address"),
            "paired": self.t("connection.bluetooth.column.paired"),
            "present": self.t("connection.bluetooth.column.present"),
            "connected": self.t("connection.bluetooth.column.connected"),
            "signal": self.t("connection.bluetooth.column.signal"),
        }
        widths = {"name": 280, "address": 175, "paired": 82, "present": 82, "connected": 92, "signal": 68}
        for column in columns:
            self.bt_tree.heading(column, text=headings[column])
            self.bt_tree.column(column, width=widths[column], minwidth=58, anchor="w", stretch=column in {"name", "address"})
        self.bt_tree.grid(row=0, column=0, sticky="nsew")
        bt_scroll = ttk.Scrollbar(table, orient="vertical", command=self.bt_tree.yview)
        bt_scroll.grid(row=0, column=1, sticky="ns")
        self.bt_tree.configure(yscrollcommand=bt_scroll.set)
        self.bt_tree.bind("<Double-1>", lambda _event: self.connect_bluetooth_selected())

        self.serial_card = Card(host)
        ttk.Label(self.serial_card, text=f"2. {self.t('connection.connect')}", style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(
            self.serial_card, text=self.t("connection.serial.help"), style="CardBody.TLabel", wraplength=980
        ).pack(anchor="w", pady=(5, 10))
        self.endpoint_combo = ttk.Combobox(self.serial_card, textvariable=self.endpoint_var, state="normal")
        self.endpoint_combo.pack(fill="x")
        serial_actions = ttk.Frame(self.serial_card, style="Card.TFrame")
        serial_actions.pack(fill="x", pady=(10, 0))
        self.serial_refresh_button = ttk.Button(serial_actions, text=self.t("action.refresh"), command=self.refresh_serial_endpoints)
        self.serial_refresh_button.pack(side="left")
        self.serial_connect_button = ttk.Button(
            serial_actions, text=self.t("action.connect"), command=self.toggle_serial_connection, style="Primary.TButton"
        )
        self.serial_connect_button.pack(side="left", padx=6)
        self.serial_connection_pill = StatusPill(serial_actions)
        self.serial_connection_pill.pack(side="left", padx=(6, 0))

        self.next_card = Card(host, style="AccentCard.TFrame")
        left = ttk.Frame(self.next_card, style="AccentCard.TFrame")
        left.pack(side="left", fill="x", expand=True)
        ttk.Label(left, text=f"3. {self.t('connection.next')}", style="AccentCardTitle.TLabel").pack(anchor="w")
        ttk.Label(left, text=self.t("connection.next.help"), style="AccentCardBody.TLabel", wraplength=780).pack(anchor="w", pady=(3, 0))
        ttk.Button(
            self.next_card, text=self.t("action.go_monitor"),
            command=lambda: self.app.navigate(WorkflowStep.MONITOR), style="Primary.TButton"
        ).pack(side="right", padx=(14, 0))

    def _mode_changed(self, _event=None) -> None:
        mode = self._selected_mode()
        self.app.settings.connection_mode = mode.value
        self.app.settings.save()
        self._apply_mode_ui(initial=False)

    def _apply_mode_ui(self, initial: bool = False) -> None:
        mode = self._selected_mode()
        self.bt_card.pack_forget()
        self.serial_card.pack_forget()
        self.next_card.pack_forget()
        if mode is TransportMode.BLUETOOTH:
            self.bt_card.pack(fill="both", expand=True, pady=(0, 12))
            self.next_card.pack(fill="x")
            if not initial:
                self.after_idle(self.search_bluetooth)
        else:
            self._bt_worker.stop_scan()
            self._bt_scan_active = False
            self.serial_card.pack(fill="x", pady=(0, 12))
            self.next_card.pack(fill="x")
            self.refresh_serial_endpoints()

    # ------------------------------------------------------------------
    # Bluetooth discovery and selection flow
    # ------------------------------------------------------------------

    def search_bluetooth(self) -> None:
        if self._hidden or self._selected_mode() is not TransportMode.BLUETOOTH:
            return
        # Discovery lifecycle belongs exclusively to the persistent Bluetooth worker.
        self.status_var.set(self.t("connection.bluetooth.status.starting"))
        self.bt_connection_pill.set_state(self.t("connection.bluetooth.status.searching"), "warn")
        self._bt_worker.submit(self._bt_worker.start_scan())

    def _clear_bluetooth_devices(self) -> None:
        for iid in self.bt_tree.get_children():
            self.bt_tree.delete(iid)
        self._bt_device_rows.clear()
        self._bt_row_devices.clear()
        self._bt_row_counter = 0

    def _upsert_bluetooth_device(self, device: dict) -> None:
        device_id = str(device.get("id", ""))
        if not device_id:
            return
        paired = self.t("common.yes") if device.get("paired") else self.t("common.no")
        present_value = device.get("present")
        present = self.t("common.unknown") if present_value is None else (self.t("common.yes") if present_value else self.t("common.no"))
        connected = self.t("common.yes") if device.get("connected") else self.t("common.no")
        signal = "—" if device.get("signal") is None else str(device.get("signal"))
        values = (device.get("name", ""), device.get("address", ""), paired, present, connected, signal)
        iid = self._bt_device_rows.get(device_id)
        if iid is None:
            iid = f"d{self._bt_row_counter}"
            self._bt_row_counter += 1
            self._bt_device_rows[device_id] = iid
            # The table stores only the DeviceInformation ID; WinRT objects stay
            # actual DeviceInformation object remains owned by BluetoothWorker.
            self._bt_row_devices[iid] = device_id
            self.bt_tree.insert("", "end", iid=iid, values=values)
        else:
            if self.bt_tree.exists(iid):
                self.bt_tree.item(iid, values=values)

    def _remove_bluetooth_device(self, device_id: str) -> None:
        iid = self._bt_device_rows.pop(device_id, None)
        if iid is None:
            return
        self._bt_row_devices.pop(iid, None)
        if self.bt_tree.exists(iid):
            self.bt_tree.delete(iid)

    def _status_text(self, payload) -> str:
        if isinstance(payload, dict) and payload.get("key"):
            return self.t(payload["key"], **payload.get("values", {}))
        return str(payload or "")

    def _poll_bluetooth_events(self) -> None:
        if self._hidden:
            return
        try:
            while True:
                event, data = self._bt_events.get_nowait()
                if event == "clear_devices":
                    self._clear_bluetooth_devices()
                elif event == "device_upsert":
                    self._upsert_bluetooth_device(data)
                elif event == "device_removed":
                    self._remove_bluetooth_device(str(data))
                elif event == "scan_started":
                    self._bt_scan_active = True
                    self.status_var.set(self.t("connection.bluetooth.status.active"))
                    self.bt_connection_pill.set_state(self.t("connection.bluetooth.status.searching"), "warn")
                elif event == "scan_stopped":
                    self._bt_scan_active = False
                    if not self.app.controller.snapshot().connected:
                        self.status_var.set(self.t("connection.bluetooth.status.stopped"))
                        self.bt_connection_pill.set_state(self.t("status.disconnected"), "idle")
                elif event == "status":
                    self.status_var.set(self._status_text(data))
                elif event == "connecting":
                    self._connect_running = True
                    self.status_var.set(self.t("connection.bluetooth.connecting", device=str(data)))
                    self.bt_connection_pill.set_state(self.t("connection.bluetooth.status.validating"), "warn")
                elif event == "thinkgear_probe":
                    self.status_var.set(
                        self.t(
                            "connection.bluetooth.status.probe",
                            bytes=data.get("bytes", 0),
                            sync=data.get("sync", 0),
                            packets=data.get("packets", 0),
                            errors=data.get("checksum_errors", 0),
                        )
                    )
                elif event == "bluetooth_connected":
                    self.status_var.set(
                        self.t("connection.bluetooth.status.rfcomm_connected", device=data.get("name", "MindFlex"))
                    )
                elif event == "mindflex_confirmed":
                    self._connect_running = False
                    self.bt_search_button.configure(state="normal")
                    self.bt_connect_button.configure(state="normal")
                    self.bt_disconnect_button.configure(state="normal")
                    endpoint = data.get("endpoint") if isinstance(data, dict) else None
                    if isinstance(endpoint, EndpointInfo):
                        self.complete_connection(endpoint)
                elif event == "connect_failed":
                    self._connect_running = False
                    self.bt_search_button.configure(state="normal")
                    self.bt_connect_button.configure(state="normal")
                    self.bt_disconnect_button.configure(state="normal")
                    self.status_var.set(self.t("connection.bluetooth.connect_error", error=str(data)))
                    self.bt_connection_pill.set_state(self.t("diag.status.failed"), "error")
                    messagebox.showwarning("Bluetooth", str(data))
                elif event == "disconnected":
                    self._connect_running = False
                    self.bt_search_button.configure(state="normal")
                    self.bt_connect_button.configure(state="normal")
                    self.bt_disconnect_button.configure(state="normal")
                    self.status_var.set(self.t("connection.disconnected"))
                    self.bt_connection_pill.set_state(self.t("status.disconnected"), "idle")
                    self.app.refresh_runtime()
                elif event == "error":
                    self.status_var.set(self.t("connection.bluetooth.scan_error", error=str(data)))
                    self.bt_connection_pill.set_state(self.t("diag.status.failed"), "error")
                    messagebox.showerror("MindFlex", str(data))
                elif event == "debug":
                    pass
        except queue.Empty:
            pass
        if not self._hidden:
            self._bt_poll_job = self.after(60, self._poll_bluetooth_events)

    def connect_bluetooth_selected(self) -> None:
        selected = self.bt_tree.selection()
        if not selected:
            messagebox.showinfo("Bluetooth", self.t("connection.bluetooth.select"))
            return
        iid = selected[0]
        device_id = self._bt_row_devices.get(iid)
        if not device_id:
            messagebox.showwarning("Bluetooth", self.t("connection.bluetooth.unavailable"))
            return
        values = self.bt_tree.item(iid, "values")
        name = values[0] if values else "MindFlex"
        self.status_var.set(self.t("connection.bluetooth.connecting", device=name))
        self.bt_connection_pill.set_state(self.t("connection.bluetooth.status.validating"), "warn")
        # Connection uses the selected DeviceInformation ID; no endpoint reconstruction.
        # here.  The same persistent worker receives the selected DeviceInfo ID
        # and performs stop-scan -> pair -> RFCOMM -> ThinkGear confirmation.
        self._bt_worker.submit(self._bt_worker.connect(device_id))

    def disconnect(self) -> None:
        if self._selected_mode() is TransportMode.BLUETOOTH or self._bt_worker.current_endpoint is not None:
            self._bt_worker.submit(self._bt_worker.disconnect())
        else:
            self.app.controller.disconnect()
        self.status_var.set(self.t("connection.disconnected"))
        self.bt_connection_pill.set_state(self.t("status.disconnected"), "idle")
        self.serial_connection_pill.set_state(self.t("status.disconnected"), "idle")
        self.app.refresh_runtime()

    # ------------------------------------------------------------------
    # Separate Serial/USB path
    # ------------------------------------------------------------------

    def refresh_serial_endpoints(self) -> None:
        endpoints = discover_endpoints(TransportMode.SERIAL)
        self._endpoint_by_display = {endpoint.display_name: endpoint for endpoint in endpoints}
        values = list(self._endpoint_by_display)
        self.endpoint_combo.configure(values=values)
        saved_id = self.app.settings.connection_endpoint
        selected = next((display for display, endpoint in self._endpoint_by_display.items() if endpoint.id == saved_id), "")
        if not selected and saved_id:
            saved_endpoint = endpoint_from_id(saved_id)
            if saved_endpoint is not None and saved_endpoint.kind.value == "serial":
                selected = saved_endpoint.display_name
                self._endpoint_by_display[selected] = saved_endpoint
                self.endpoint_combo.configure(values=list(self._endpoint_by_display))
        if selected:
            self.endpoint_var.set(selected)
        elif self.endpoint_var.get() not in values:
            self.endpoint_var.set(values[0] if values else "")
        self.status_var.set(self.t("connection.devices_found", count=len(values), bluetooth=0))

    def _resolve_serial_endpoint(self) -> EndpointInfo:
        value = self.endpoint_var.get().strip()
        endpoint = self._endpoint_by_display.get(value)
        if endpoint is not None:
            return endpoint
        return manual_endpoint(value, TransportMode.SERIAL)

    def toggle_serial_connection(self) -> None:
        snap = self.app.controller.snapshot()
        if snap.connected:
            self.disconnect()
            return
        if not self.endpoint_var.get().strip():
            self.status_var.set(self.t("connection.select_device"))
            return
        try:
            endpoint = self._resolve_serial_endpoint()
            self.app.controller.connect(endpoint, self.app.settings.baudrate)
        except Exception as exc:
            self.status_var.set(self.t("connection.error", error=str(exc)))
            return
        self.complete_connection(endpoint)

    def complete_connection(self, endpoint: EndpointInfo) -> None:
        self.app.settings.connection_mode = endpoint.kind.value
        self.app.settings.connection_endpoint = endpoint.id
        self.app.settings.save()
        self.mode_var.set(self._mode_label(endpoint.kind.value))
        self.status_var.set(self.t("connection.mindflex_confirmed", device=endpoint.description or endpoint.address))
        if endpoint.kind.value == "bluetooth":
            self.bt_connection_pill.set_state(self.t("connection.bluetooth.status.confirmed"), "ok")
        else:
            self.serial_connection_pill.set_state(self.t("status.receiving"), "ok")
        self.app.refresh_runtime()
        # Move to Monitor once the transport is validated and the controller is connected.
        self.after(250, lambda: self.app.navigate(WorkflowStep.MONITOR))

    def start_simulator(self) -> None:
        self._bt_worker.stop_scan()
        if self._bt_worker.current_endpoint is not None:
            future = self._bt_worker.submit(self._bt_worker.disconnect())
            if future is not None:
                try:
                    future.result(timeout=2.5)
                except Exception:
                    pass
        self.app.controller.start_simulation(self.app.settings.raw_sample_rate)
        self.status_var.set(self.t("connection.simulator.started"))
        self.app.refresh_runtime()
        self.after(250, lambda: self.app.navigate(WorkflowStep.MONITOR))

    def update_snapshot(self, snap: EEGSnapshot) -> None:
        if snap.connected:
            if snap.transport == "bluetooth":
                state = "ok" if snap.receiving else "warn"
                text = self.t("status.receiving") if snap.receiving else self.t("status.connected_waiting")
                self.bt_connection_pill.set_state(text, state)
            elif snap.transport == "serial":
                state = "ok" if snap.receiving else "warn"
                text = self.t("status.receiving") if snap.receiving else self.t("status.connected_waiting")
                self.serial_connection_pill.set_state(text, state)
        else:
            if not self._bt_scan_active and not self._connect_running:
                self.bt_connection_pill.set_state(self.t("status.disconnected"), "idle")
            self.serial_connection_pill.set_state(self.t("status.disconnected"), "idle")
        if snap.error and not self._connect_running:
            self.status_var.set(self.t("connection.error", error=snap.error))

    def on_hide(self) -> None:
        self._hidden = True
        if self._bt_poll_job is not None:
            try:
                self.after_cancel(self._bt_poll_job)
            except tk.TclError:
                pass
            self._bt_poll_job = None
        # The application owns the worker; page navigation never closes transport.


class MonitorView(ttk.Frame):
    def __init__(self, master, app: "MindFlexApp") -> None:
        super().__init__(master, style="Page.TFrame", padding=24)
        self.app = app
        self.tr = app.tr
        self._visible = False
        self._recorder: SessionRecorder | None = None
        self.recording_var = tk.StringVar(value=self.t("monitor.recording.idle"))
        self.metric_vars = {
            name: tk.StringVar(value="—")
            for name in ("contact_quality", "raw_rate", "attention", "meditation", "packets")
        }
        self.band_vars = {name: tk.StringVar(value="—") for name in EEG_BANDS}
        self._build()
        self.scheduler = FrameScheduler(self, self._render_graphs, fps=min(20, app.settings.graph_fps))

    def t(self, key, **values):
        return self.tr.t(key, **values)

    def _build(self) -> None:
        PageHeader(self, self.t("monitor.title"), self.t("monitor.subtitle")).pack(fill="x")
        scroll = ScrollableFrame(self)
        scroll.pack(fill="both", expand=True, pady=(18, 0))
        host = scroll.body
        self._page_scroll = scroll

        metrics = ResponsiveCardGrid(host, min_card_width=155, max_columns=5, gap=10)
        metrics.pack(fill="x", pady=(0, 12))
        specs = (
            ("contact_quality", "monitor.contact_quality"),
            ("raw_rate", "monitor.raw_rate"),
            ("attention", "monitor.attention"),
            ("meditation", "monitor.meditation"),
            ("packets", "monitor.packets"),
        )
        for name, label_key in specs:
            card = Card(metrics, padding=14)
            ttk.Label(card, text=self.t(label_key), style="MetricLabel.TLabel").pack(anchor="w")
            ttk.Label(card, textvariable=self.metric_vars[name], style="MetricLarge.TLabel").pack(anchor="w", pady=(3, 0))
            metrics.add(card)

        content = ResponsiveCardGrid(host, min_card_width=430, max_columns=2, gap=14)
        content.pack(fill="x")

        raw_card = Card(content, padding=10)
        ttk.Label(raw_card, text=self.t("monitor.raw"), style="CardTitle.TLabel").pack(anchor="w", padx=8, pady=(4, 0))
        self.raw_plot = FastLinePlot(raw_card, xlabel=self.t("axis.seconds"), ylabel=self.t("axis.raw"))
        self.raw_plot.widget.configure(height=280)
        self.raw_plot.widget.pack(fill="both", expand=True)
        content.add(raw_card)

        cognitive_card = Card(content, padding=10)
        ttk.Label(cognitive_card, text=self.t("monitor.cognitive"), style="CardTitle.TLabel").pack(anchor="w", padx=8, pady=(4, 0))
        self.metric_plot = FastMultiLinePlot(cognitive_card, ("attention", "meditation"), xlabel=self.t("axis.seconds"), ylabel="%", ylim=(0, 100))
        self.metric_plot.set_labels(
            self.t("axis.seconds"), "%",
            {"attention": self.t("monitor.attention"), "meditation": self.t("monitor.meditation")},
        )
        self.metric_plot.widget.configure(height=250)
        self.metric_plot.widget.pack(fill="both", expand=True)
        content.add(cognitive_card)

        bands = Card(content, padding=14)
        ttk.Label(bands, text=self.t("monitor.bands"), style="CardTitle.TLabel").pack(anchor="w", pady=(0, 8))
        band_grid = ttk.Frame(bands, style="Card.TFrame")
        band_grid.pack(fill="x")
        for idx, name in enumerate(EEG_BANDS):
            row, col = divmod(idx, 2)
            cell = ttk.Frame(band_grid, style="Card.TFrame")
            cell.grid(row=row, column=col, sticky="ew", padx=(0 if col == 0 else 10, 10 if col == 0 else 0), pady=3)
            band_grid.columnconfigure(col, weight=1)
            ttk.Label(cell, text=self.t(f"band.{name}"), style="CardBody.TLabel").pack(side="left")
            ttk.Label(cell, textvariable=self.band_vars[name], style="CardMetaStrong.TLabel").pack(side="right")
        ttk.Separator(bands).pack(fill="x", pady=12)
        ttk.Label(bands, text=self.t("monitor.recording"), style="CardBodyStrong.TLabel").pack(anchor="w")
        self.record_button = ttk.Button(bands, text=self.t("monitor.recording.start"), command=self._toggle_recording)
        self.record_button.pack(fill="x", pady=(6, 0))
        ttk.Label(bands, textvariable=self.recording_var, style="CardMeta.TLabel", wraplength=540).pack(anchor="w", pady=(6, 10))
        ttk.Separator(bands).pack(fill="x", pady=(0, 12))
        ttk.Button(bands, text=self.t("action.next_neuro"), command=lambda: self.app.navigate(WorkflowStep.NEURO_CONTROL), style="Primary.TButton").pack(fill="x")
        content.add(bands)

    def _toggle_recording(self) -> None:
        if self._recorder is not None:
            self._stop_recording()
            return
        if not self.app.controller.snapshot().receiving:
            self.recording_var.set(self.t("monitor.recording.stream_required"))
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), (self.t("file.all"), "*.*")],
        )
        if not path:
            return
        recorder = SessionRecorder(Path(path))
        try:
            recorder.start()
        except OSError as exc:
            self.recording_var.set(self.t("monitor.recording.error", error=str(exc)))
            return
        self._recorder = recorder
        self.app.controller.add_listener(self._record_snapshot)
        self.record_button.configure(text=self.t("monitor.recording.stop"))
        self.recording_var.set(self.t("monitor.recording.active", path=path))

    def _record_snapshot(self, snap: EEGSnapshot) -> None:
        recorder = self._recorder
        if recorder is not None and snap.receiving and snap.timestamp:
            recorder.append(snap)

    def _stop_recording(self) -> None:
        recorder, self._recorder = self._recorder, None
        self.app.controller.remove_listener(self._record_snapshot)
        if recorder is not None:
            recorder.stop()
        if hasattr(self, "record_button"):
            self.record_button.configure(text=self.t("monitor.recording.start"))
        self.recording_var.set(self.t("monitor.recording.stopped"))

    def update_snapshot(self, snap: EEGSnapshot) -> None:
        if self._recorder is not None and self._recorder.error:
            error = self._recorder.error
            self._stop_recording()
            self.recording_var.set(self.t("monitor.recording.error", error=error))
        if snap.poor_signal is None or snap.signal_age > 4.0:
            self.metric_vars["contact_quality"].set("—")
        else:
            contact_quality = max(0, min(100, round((200 - snap.poor_signal) / 2)))
            self.metric_vars["contact_quality"].set(f"{contact_quality}%")
        self.metric_vars["raw_rate"].set("—" if snap.raw_total_samples == 0 else f"{snap.raw_rate_hz:.0f} Hz")
        source_label = {"thinkgear": "TGAM", "raw": "RAW*"}
        attention_source = f" · {source_label.get(snap.attention_source, '')}" if snap.attention_source else ""
        meditation_source = f" · {source_label.get(snap.meditation_source, '')}" if snap.meditation_source else ""
        self.metric_vars["attention"].set("—" if snap.attention is None else f"{snap.attention}%{attention_source}")
        self.metric_vars["meditation"].set("—" if snap.meditation is None else f"{snap.meditation}%{meditation_source}")
        self.metric_vars["packets"].set(str(snap.packets))
        for name in EEG_BANDS:
            value = snap.bands.get(name)
            self.band_vars[name].set("—" if value is None else f"{value:,}".replace(",", " "))

    def _render_graphs(self) -> None:
        if not self._visible:
            return
        seconds = self.app.settings.graph_window_seconds
        x, y = self.app.controller.raw_window(seconds, max_points=self.app.settings.max_plot_points)
        if x:
            self.raw_plot.update(x, y, xlim=(-seconds, 0.0), robust_ylim=True)
        data = {}
        for name in ("attention", "meditation"):
            mx, my = self.app.controller.metric_window(name, max(20.0, seconds * 3), max_points=900)
            data[name] = (mx, my)
        self.metric_plot.update(data, xlim=(-max(20.0, seconds * 3), 0.0), ylim=(0, 100))

    def on_show(self) -> None:
        self._visible = True
        self.scheduler.start()

    def on_hide(self) -> None:
        self._visible = False
        self.scheduler.stop()
        if self._recorder is not None:
            self._stop_recording()


class DiagnosticsView(ttk.Frame):
    def __init__(self, master, app: "MindFlexApp") -> None:
        super().__init__(master, style="Page.TFrame", padding=24)
        self.app = app
        self.tr = app.tr
        self.sequence = DiagnosticSequence(app.controller, app.rules.policy)
        self.detail_var = tk.StringVar(value=self.t("diagnostics.ready"))
        self._build()

    def t(self, key, **values):
        return self.tr.t(key, **values)

    def _build(self) -> None:
        PageHeader(self, self.t("diagnostics.title"), self.t("diagnostics.subtitle")).pack(fill="x")
        scroll = ScrollableFrame(self)
        scroll.pack(fill="both", expand=True, pady=(18, 0))
        host = scroll.body
        self._page_scroll = scroll

        body = ResponsiveCardGrid(host, min_card_width=360, max_columns=2, gap=14)
        body.pack(fill="x")

        list_card = Card(body)
        ttk.Label(list_card, text=self.t("diagnostics.sequence"), style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(list_card, text=self.t("diagnostics.sequence.help"), style="CardBody.TLabel", wraplength=720).pack(anchor="w", pady=(5, 12))
        self.rows_host = ttk.Frame(list_card, style="Card.TFrame")
        self.rows_host.pack(fill="x")
        actions = ttk.Frame(list_card, style="Card.TFrame")
        actions.pack(fill="x", pady=(14, 0))
        ttk.Button(actions, text=self.t("diagnostics.run_next"), command=self.run_next, style="Primary.TButton").pack(side="left")
        ttk.Button(actions, text=self.t("diagnostics.run_all"), command=self.run_all).pack(side="left", padx=(6, 0))
        ttk.Button(actions, text=self.t("action.reset"), command=self.reset).pack(side="left", padx=(6, 0))
        body.add(list_card)

        detail = Card(body, style="AccentCard.TFrame")
        ttk.Label(detail, text=self.t("diagnostics.current"), style="AccentCardTitle.TLabel").pack(anchor="w")
        ttk.Label(detail, textvariable=self.detail_var, style="AccentCardBody.TLabel", wraplength=720, justify="left").pack(fill="x", anchor="w", pady=(8, 0))
        body.add(detail)
        self._render_results()

    def _render_results(self) -> None:
        clear_children(self.rows_host)
        for idx, result in enumerate(self.sequence.results):
            row = ttk.Frame(self.rows_host, style="Card.TFrame", padding=(0, 7))
            row.pack(fill="x")
            status_text = self.t(f"diag.status.{result.status.value}")
            ttk.Label(row, text=f"{idx + 1}", style="DiagNumber.TLabel", width=3).pack(side="left")
            ttk.Label(row, text=self.t(f"diag.test.{result.key}"), style="CardBodyStrong.TLabel").pack(side="left", fill="x", expand=True)
            ttk.Label(row, text=status_text, style=f"Diag{result.status.value.title()}.TLabel").pack(side="right")
            if idx < len(self.sequence.results) - 1:
                ttk.Separator(self.rows_host).pack(fill="x")

    def run_next(self) -> None:
        result = self.sequence.run_next()
        if result is None:
            self.detail_var.set(self.t("diagnostics.complete"))
        else:
            detail = self.t(result.detail_key) if result.detail_key else ""
            self.detail_var.set(self.t("diagnostics.detail", detail=detail, value=result.value))
        self._render_results()

    def run_all(self) -> None:
        results = self.sequence.run_all()
        failures = [result for result in results if result.status == DiagnosticStatus.FAILED]
        warnings = [result for result in results if result.status == DiagnosticStatus.WARNING]
        if failures:
            self.detail_var.set(self.t("diagnostics.complete_failures", count=len(failures)))
        elif warnings:
            self.detail_var.set(self.t("diagnostics.complete_warnings", count=len(warnings)))
        elif self.sequence.complete:
            self.detail_var.set(self.t("diagnostics.complete"))
        self._render_results()

    def reset(self) -> None:
        self.sequence.reset()
        self.detail_var.set(self.t("diagnostics.ready"))
        self._render_results()

class MindFlexApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.withdraw()
        self.settings = Settings.load()
        self.tr = Translator(self.settings.language)
        self._configure_style()
        # A modal whose parent is withdrawn can remain invisible on Windows.
        # Show and raise the root before asking for the initial person/profile;
        # the application shell replaces this blank root immediately afterward.
        self.title(APP_NAME)
        self.geometry("1280x820")
        self.minsize(1024, 680)
        self.deiconify()
        self.update_idletasks()
        self.lift()
        self.user_profile = self._select_user_profile()
        self.user_profile.ensure_storage()
        configure_plot_font(self.tr.language)
        self.rules = WorkflowRules()
        self.controller = EEGController(raw_sample_rate=self.settings.raw_sample_rate)
        # RFCOMM transport is isolated from parsing, signal processing and rendering.
        # The worker returns to DataReader immediately; controller ingestion is queued.
        self.bluetooth_events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.bluetooth_worker = BluetoothWorker(
            self.bluetooth_events,
            on_connecting=self._bluetooth_connecting,
            on_connected=self._bluetooth_connected,
            on_bytes=self.controller.feed_bluetooth_bytes,
            on_disconnected=self._bluetooth_disconnected,
            on_error=self._bluetooth_error,
        )
        self.bluetooth_worker.start()
        self._bluetooth_initial_scan_started = False
        self.neuro_runtime = NeuroRuntime(
            user_name=self.user_profile.full_name,
            model_store=ModelStore(self.user_profile.data_dir, owner_name=self.user_profile.full_name),
        )
        self.current_step = WorkflowStep.CONNECTION
        self.current_view = None
        self._poll_job = None
        self._last_runtime_key = None
        self.title(APP_NAME)
        self.geometry("1280x820")
        self.minsize(1024, 680)
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.tr.add_listener(self._language_rebuild)
        self._build_shell()
        self.deiconify()
        self.after(650, self._start_initial_bluetooth_scan)
        self._poll()

    def _ask_user_profile(self, parent=None, *, cancel_exits: bool = True) -> UserProfile | None:
        owner = parent or self
        while True:
            value = simpledialog.askstring(
                self.tr.t("user.prompt.title"),
                self.tr.t("user.prompt.message"),
                parent=owner,
            )
            if value is None:
                if not cancel_exits:
                    return None
                self.destroy()
                raise SystemExit(0)
            try:
                return UserProfile.from_full_name(value)
            except ValueError:
                messagebox.showwarning(
                    self.tr.t("user.prompt.title"),
                    self.tr.t("user.prompt.required"),
                    parent=owner,
                )

    def _select_user_profile(self) -> UserProfile:
        profiles = UserProfile.list_saved()
        if not profiles:
            return self._ask_user_profile()

        selected: list[UserProfile] = []
        dialog = tk.Toplevel(self)
        dialog.title(self.tr.t("user.select.title"))
        dialog.geometry("520x430")
        dialog.minsize(420, 340)
        dialog.transient(self)
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)

        host = ttk.Frame(dialog, padding=24)
        host.pack(fill="both", expand=True)
        ttk.Label(host, text=self.tr.t("user.select.title"), style="PageTitle.TLabel").pack(anchor="w")
        ttk.Label(
            host,
            text=self.tr.t("user.select.message"),
            style="CardBody.TLabel",
            wraplength=460,
        ).pack(anchor="w", pady=(6, 16))

        list_frame = ttk.Frame(host)
        list_frame.pack(fill="both", expand=True)
        profile_list = tk.Listbox(list_frame, activestyle="none", font=("Segoe UI", 11), exportselection=False)
        profile_list.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=profile_list.yview)
        scrollbar.pack(side="right", fill="y")
        profile_list.configure(yscrollcommand=scrollbar.set)
        for profile in profiles:
            profile_list.insert("end", profile.full_name)
        profile_list.selection_set(0)
        profile_list.activate(0)

        def use_selected(_event=None) -> None:
            selection = profile_list.curselection()
            if not selection:
                return
            selected.append(profiles[int(selection[0])])
            dialog.destroy()

        def delete_selected() -> None:
            selection = profile_list.curselection()
            if not selection or not profiles:
                return
            index = int(selection[0])
            profile = profiles[index]
            confirmed = messagebox.askyesno(
                self.tr.t("user.delete.title"),
                self.tr.t("user.delete.message", name=profile.full_name),
                parent=dialog,
                icon="warning",
            )
            if not confirmed:
                return
            try:
                profile.delete_storage()
            except (OSError, ValueError) as exc:
                messagebox.showerror(
                    self.tr.t("user.delete.title"),
                    self.tr.t("user.delete.error", error=str(exc)),
                    parent=dialog,
                )
                return
            profiles.pop(index)
            profile_list.delete(index)
            if profiles:
                next_index = min(index, len(profiles) - 1)
                profile_list.selection_set(next_index)
                profile_list.activate(next_index)
            else:
                profile_list.insert("end", self.tr.t("user.select.empty"))
                profile_list.configure(state="disabled")

        def create_new() -> None:
            profile = self._ask_user_profile(dialog, cancel_exits=False)
            if profile is None:
                return
            profile.ensure_storage()
            selected.append(profile)
            dialog.destroy()

        actions = ttk.Frame(host)
        actions.pack(fill="x", pady=(16, 0))
        ttk.Button(actions, text=self.tr.t("user.new.button"), command=create_new).pack(side="left")
        ttk.Button(actions, text=self.tr.t("user.delete.button"), command=delete_selected).pack(side="left", padx=(8, 0))
        ttk.Button(
            actions,
            text=self.tr.t("user.select.button"),
            command=use_selected,
            style="Primary.TButton",
        ).pack(side="right")
        profile_list.bind("<Double-1>", use_selected)
        dialog.bind("<Return>", use_selected)
        dialog.update_idletasks()
        dialog.lift()
        dialog.focus_force()
        dialog.grab_set()
        self.wait_window(dialog)
        if not selected:
            self.destroy()
            raise SystemExit(0)
        return selected[0]

    def t(self, key, **values):
        return self.tr.t(key, **values)

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        bg = "#f5f7fb"
        card = "#ffffff"
        fg = "#18202a"
        muted = "#657184"
        soft_accent = "#eaf1ff"
        ok = "#16794b"
        warn = "#9a6700"
        error = "#b42318"
        self.configure(bg=bg)
        base_font = ("Segoe UI", 10)
        style.configure(".", font=base_font, background=bg, foreground=fg)
        style.configure("Page.TFrame", background=bg)
        style.configure("Card.TFrame", background=card, relief="flat")
        style.configure("AccentCard.TFrame", background=soft_accent, relief="flat")
        style.configure("Header.TFrame", background=card)
        style.configure("Sidebar.TFrame", background="#111827")
        style.configure("PageTitle.TLabel", background=bg, foreground=fg, font=("Segoe UI Semibold", 22))
        style.configure("PageSubtitle.TLabel", background=bg, foreground=muted, font=("Segoe UI", 10))
        style.configure("SectionTitle.TLabel", background=bg, foreground=fg, font=("Segoe UI Semibold", 14))
        style.configure("SectionSubtitle.TLabel", background=bg, foreground=muted)
        style.configure("CardTitle.TLabel", background=card, foreground=fg, font=("Segoe UI Semibold", 12))
        style.configure("CardBody.TLabel", background=card, foreground=muted)
        style.configure("CardBodyStrong.TLabel", background=card, foreground=fg, font=("Segoe UI Semibold", 10))
        style.configure("CardMeta.TLabel", background=card, foreground=muted, font=("Segoe UI", 9))
        style.configure("CardMetaStrong.TLabel", background=card, foreground=fg, font=("Segoe UI Semibold", 9))
        style.configure("AccentCardTitle.TLabel", background=soft_accent, foreground=fg, font=("Segoe UI Semibold", 12))
        style.configure("AccentCardBody.TLabel", background=soft_accent, foreground=muted)
        style.configure("AccentCardMetric.TLabel", background=soft_accent, foreground=fg, font=("Segoe UI Semibold", 18))
        style.configure("MetricLabel.TLabel", background=card, foreground=muted, font=("Segoe UI", 9))
        style.configure("MetricLarge.TLabel", background=card, foreground=fg, font=("Segoe UI Semibold", 18))
        style.configure("MetricHuge.TLabel", background=card, foreground=fg, font=("Segoe UI Semibold", 30))
        style.configure("InlineStatus.TLabel", background=bg, foreground=muted)
        style.configure("Primary.TButton", font=("Segoe UI Semibold", 10), padding=(14, 8))
        style.configure("Step.TButton", padding=(8, 8))
        style.configure("StepCurrent.TButton", padding=(8, 8), font=("Segoe UI Semibold", 10))
        style.configure("StepHint.TLabel", background=bg, foreground=muted, font=("Segoe UI", 8))
        style.configure("SidebarTitle.TLabel", background="#111827", foreground="#ffffff", font=("Segoe UI Semibold", 13))
        style.configure("SidebarHint.TLabel", background="#111827", foreground="#9ca3af", font=("Segoe UI", 9))
        style.configure("Sidebar.TButton", background="#111827", foreground="#e5e7eb", anchor="w", padding=(14, 11), borderwidth=0)
        style.configure("SidebarCurrent.TButton", background="#1f2937", foreground="#ffffff", anchor="w", padding=(14, 11), borderwidth=0, font=("Segoe UI Semibold", 10))
        style.map("Sidebar.TButton", background=[("active", "#1f2937")])
        style.map("SidebarCurrent.TButton", background=[("active", "#374151")])
        for name, color, background in (
            ("StatusOk", ok, "#eaf8f0"), ("StatusWarn", warn, "#fff7df"),
            ("StatusError", error, "#fff0ee"), ("StatusIdle", muted, "#eef1f5"),
        ):
            style.configure(f"{name}.TLabel", foreground=color, background=background, padding=(9, 4), font=("Segoe UI Semibold", 9))
        style.configure("DiagNumber.TLabel", background=card, foreground=muted, font=("Segoe UI Semibold", 9))
        for status, color in (("Pending", muted), ("Passed", ok), ("Warning", warn), ("Failed", error)):
            style.configure(f"Diag{status}.TLabel", background=card, foreground=color, font=("Segoe UI Semibold", 9))

    def _build_shell(self) -> None:
        clear_children(self)
        header = ttk.Frame(self, style="Header.TFrame", padding=(20, 12))
        header.pack(fill="x")
        ttk.Label(header, text=APP_NAME, style="CardTitle.TLabel").pack(side="left")
        ttk.Label(
            header,
            text=self.t("user.active", name=self.user_profile.full_name),
            style="CardMetaStrong.TLabel",
        ).pack(side="left", padx=(14, 0))
        self.global_status = StatusPill(header)
        self.global_status.pack(side="left", padx=(16, 0))
        ttk.Button(header, text=self.t("about.button"), command=self.show_about).pack(side="right")
        language_names = list(LANGUAGES.values())
        current_name = LANGUAGES.get(self.tr.language, LANGUAGES["en"])
        self.language_var = tk.StringVar(value=current_name)
        combo = ttk.Combobox(header, textvariable=self.language_var, values=language_names, state="readonly", width=20)
        combo.pack(side="right", padx=(0, 8))
        combo.bind("<<ComboboxSelected>>", self._change_language)

        shell = ttk.Frame(self, style="Page.TFrame")
        shell.pack(fill="both", expand=True)
        self.sidebar = ttk.Frame(shell, style="Sidebar.TFrame", width=235, padding=(12, 20))
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)
        self.content = ttk.Frame(shell, style="Page.TFrame")
        self.content.pack(side="left", fill="both", expand=True)
        self._render_sidebar()
        self._show_current_view()

    def _render_sidebar(self) -> None:
        clear_children(self.sidebar)
        ttk.Label(self.sidebar, text=self.t("workflow.title"), style="SidebarTitle.TLabel").pack(anchor="w", padx=8)
        ttk.Label(self.sidebar, text=self.t("workflow.subtitle"), style="SidebarHint.TLabel", wraplength=190).pack(anchor="w", padx=8, pady=(3, 14))
        labels = (
            (WorkflowStep.CONNECTION, "workflow.connection"),
            (WorkflowStep.MONITOR, "workflow.monitor"),
            (WorkflowStep.NEURO_CONTROL, "workflow.neuro"),
            (WorkflowStep.DIAGNOSTICS, "workflow.diagnostics"),
        )
        state = self.runtime_state()
        for step, key in labels:
            allowed = self.rules.workflow_allowed(step, state)
            style = "SidebarCurrent.TButton" if step == self.current_step else "Sidebar.TButton"
            button = ttk.Button(
                self.sidebar,
                text=f"{int(step) + 1}.  {self.t(key)}",
                style=style,
                command=lambda s=step: self.navigate(s),
                state="normal" if allowed else "disabled",
            )
            button.pack(fill="x", pady=2)
        ttk.Separator(self.sidebar).pack(fill="x", pady=16)
        ttk.Label(self.sidebar, text=self._next_hint(), style="SidebarHint.TLabel", wraplength=190).pack(anchor="w", padx=8)

    def _next_hint(self) -> str:
        state = self.runtime_state()
        if not state.connected:
            return self.t("hint.connect_first")
        if not state.receiving:
            return self.t("hint.wait_stream")
        if not state.model_ready:
            return self.t("hint.train_model")
        if not state.model_validated:
            return self.t("hint.validate_model")
        return self.t("hint.ready")

    def runtime_state(self) -> RuntimeState:
        snap = self.controller.snapshot()
        model_ready, model_validated, live_ready, communication_ready = self.neuro_runtime.readiness_flags(self.rules)
        return RuntimeState(
            connected=snap.connected,
            receiving=snap.receiving,
            model_ready=model_ready,
            model_validated=model_validated,
            live_ready=live_ready,
            communication_ready=communication_ready,
        )

    def navigate(self, step: WorkflowStep | int) -> None:
        step = WorkflowStep(int(step))
        if not self.rules.workflow_allowed(step, self.runtime_state()):
            return
        if self.current_view is not None and hasattr(self.current_view, "on_hide"):
            self.current_view.on_hide()
        self.current_step = step
        self._render_sidebar()
        self._show_current_view()

    def _show_current_view(self) -> None:
        clear_children(self.content)
        builders = {
            WorkflowStep.CONNECTION: lambda: ConnectionView(self.content, self),
            WorkflowStep.MONITOR: lambda: MonitorView(self.content, self),
            WorkflowStep.NEURO_CONTROL: lambda: NeuroControlView(
                self.content, self.controller, self.rules, self.tr, self.neuro_runtime, self.refresh_runtime
            ),
            WorkflowStep.DIAGNOSTICS: lambda: DiagnosticsView(self.content, self),
        }
        self.current_view = builders[self.current_step]()
        self.current_view.pack(fill="both", expand=True)
        if hasattr(self.current_view, "on_show"):
            self.current_view.on_show()

    @staticmethod
    def _runtime_key(state: RuntimeState) -> tuple[bool, bool, bool, bool, bool, bool]:
        return (
            state.connected,
            state.receiving,
            state.model_ready,
            state.model_validated,
            state.live_ready,
            state.communication_ready,
        )

    def refresh_runtime(self) -> None:
        state = self.runtime_state()
        self._last_runtime_key = self._runtime_key(state)
        if not self.rules.workflow_allowed(self.current_step, state):
            if self.current_view is not None and hasattr(self.current_view, "on_hide"):
                self.current_view.on_hide()
            self.current_step = WorkflowStep.MONITOR if state.connected else WorkflowStep.CONNECTION
            self._show_current_view()
        self._render_sidebar()

    def _bluetooth_connecting(self, endpoint: EndpointInfo) -> None:
        self.controller.prepare_bluetooth_connection(endpoint)

    def _bluetooth_connected(self, endpoint: EndpointInfo) -> None:
        self.controller.confirm_bluetooth_connection(endpoint)

    def _bluetooth_disconnected(self) -> None:
        self.controller.bluetooth_disconnected()

    def _bluetooth_error(self, exc: Exception) -> None:
        self.controller.bluetooth_error(exc)

    def _poll(self) -> None:
        snap = self.controller.snapshot()
        if snap.connected and snap.receiving:
            self.global_status.set_state(self.t("status.receiving"), "ok")
        elif snap.connected:
            self.global_status.set_state(self.t("status.connected_waiting"), "warn")
        else:
            self.global_status.set_state(self.t("status.disconnected"), "idle")
        if self.current_view is not None and hasattr(self.current_view, "update_snapshot"):
            self.current_view.update_snapshot(snap)

        state = self.runtime_state()
        key = self._runtime_key(state)
        if key != self._last_runtime_key:
            self._last_runtime_key = key
            if not self.rules.workflow_allowed(self.current_step, state):
                if self.current_view is not None and hasattr(self.current_view, "on_hide"):
                    self.current_view.on_hide()
                self.current_step = WorkflowStep.MONITOR if state.connected else WorkflowStep.CONNECTION
                self._show_current_view()
            self._render_sidebar()
        self._poll_job = self.after(100, self._poll)

    def _start_initial_bluetooth_scan(self) -> None:
        if self._bluetooth_initial_scan_started:
            return
        if self.settings.connection_mode != "bluetooth":
            return
        if not self.bluetooth_worker.backend_available:
            return
        if self.controller.snapshot().connected:
            return
        self._bluetooth_initial_scan_started = True
        self.bluetooth_worker.submit(self.bluetooth_worker.start_scan())

    def _change_language(self, _event=None) -> None:
        selected = self.language_var.get()
        code = next((code for code, name in LANGUAGES.items() if name == selected), "en")
        self.settings.language = code
        self.settings.save()
        self.tr.set_language(code)

    def _language_rebuild(self) -> None:
        configure_plot_font(self.tr.language)
        if self.current_view is not None and hasattr(self.current_view, "on_hide"):
            self.current_view.on_hide()
        self._build_shell()

    def show_about(self) -> None:
        messagebox.showinfo(self.t("about.title"), f"{self.t('about.created_by')} {APP_AUTHOR} / {APP_HANDLE}")

    def _close(self) -> None:
        if self._poll_job is not None:
            try:
                self.after_cancel(self._poll_job)
            except Exception:
                pass
        if self.current_view is not None and hasattr(self.current_view, "on_hide"):
            self.current_view.on_hide()
        self.bluetooth_worker.close()
        self.controller.close()
        self.settings.save()
        self.destroy()


def run() -> None:
    app = MindFlexApp()
    app.mainloop()
