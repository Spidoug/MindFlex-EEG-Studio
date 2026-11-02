"""Fallback used on systems without a supported native Bluetooth backend."""
from __future__ import annotations

class BluetoothWorker:
    backend_available = False
    current_endpoint = None

    def __init__(self, gui_queue, **callbacks):
        self.gui_queue = gui_queue

    def start(self):
        return None

    def stop_scan(self):
        return None

    def submit(self, coroutine):
        coroutine.close()
        self.gui_queue.put(('error', 'Native Bluetooth Classic requires Windows or Linux.'))
        return None

    async def start_scan(self):
        return None

    async def connect(self, device_id):
        return None

    async def disconnect(self):
        return None

    def close(self):
        return None
