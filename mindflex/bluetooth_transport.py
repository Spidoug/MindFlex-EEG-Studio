"""Select the platform-native Bluetooth Classic transport at import time.

Windows uses WinRT; Linux uses BlueZ and native RFCOMM sockets. Neither
platform imports the other platform's Bluetooth implementation.
"""
from __future__ import annotations
import sys

if sys.platform == 'win32':
    from .bluetooth_windows import BluetoothWorker
elif sys.platform.startswith('linux'):
    from .bluetooth_linux import BluetoothWorker
else:
    from .bluetooth_unavailable import BluetoothWorker

__all__ = ['BluetoothWorker']
