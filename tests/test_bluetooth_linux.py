"""Linux BlueZ/RFCOMM contract checks using fake sockets, no headset needed."""
from __future__ import annotations

import asyncio
import queue
import socket
import sys
import unittest
from unittest.mock import AsyncMock, patch

from mindflex.bluetooth_linux import BluetoothWorker, _Device
from mindflex.parser import make_packet


@unittest.skipUnless(sys.platform.startswith('linux'), 'BlueZ is Linux-only')
class LinuxBluetoothTests(unittest.TestCase):
    def new_worker(self, **callbacks):
        w = BluetoothWorker(queue.Queue(), **callbacks)
        w.backend_available = True
        return w

    def messages(self, worker):
        messages = []
        while not worker.gui_queue.empty():
            messages.append(worker.gui_queue.get_nowait())
        return messages

    def test_platform_selection_excludes_winrt_modules(self):
        from mindflex.bluetooth_transport import BluetoothWorker as selected
        self.assertIs(selected, BluetoothWorker)
        self.assertNotIn('mindflex.bluetooth_windows', sys.modules)

    def test_device_line_parsing(self):
        text = ('Device aa:bb:cc:dd:ee:01 MindFlex EEG\n'
                '[NEW] Device 00:11:22:33:44:55 TGAM UART\n'
                '[CHG] Device AA:BB:CC:DD:EE:01 Connected: yes\n'
                'unexpected junk\n')
        self.assertEqual(BluetoothWorker._parse_devices(text), {
            'AA:BB:CC:DD:EE:01': 'MindFlex EEG',
            '00:11:22:33:44:55': 'TGAM UART',
        })

    def test_sdp_discards_non_spp_channel(self):
        services = '''Service Name: OBEX Object Push\n  Channel: 12
Service Name: Serial Port\n  Service Class ID List:\n    "Serial Port" (0x1101)\n  Channel: 7
Service Name: Other\n  Channel: 20
'''
        self.assertEqual(BluetoothWorker._serial_channels(services), [7])
        self.assertEqual(BluetoothWorker._serial_channels('Service Name: Audio\nChannel: 1'), [])

    def test_scan_lists_paired_and_discovered(self):
        async def run():
            w = self.new_worker()
            async def fake(args, timeout=4):
                if args[-1] == 'show':
                    return 0, 'Controller AB:CD\nPowered: yes'
                if args[-2:] == ['devices', 'Paired']:
                    return 0, 'Device 01:02:03:04:05:06 Paired MindFlex'
                if args[-1] == 'devices':
                    return 0, 'Device 01:02:03:04:05:06 Paired MindFlex\nDevice 02:03:04:05:06:07 TGAM'
                if 'scan' in args:
                    return 0, 'Discovery started'
                self.fail(f'Unknown command: {args}')
            w._call = fake
            with patch('mindflex.bluetooth_linux.shutil.which', return_value='/usr/bin/bluetoothctl'):
                await w.start_scan()
            events = self.messages(w)
            self.assertEqual(events[0][0], 'scan_started')
            self.assertEqual(events[-1][0], 'scan_stopped')
            self.assertEqual(set(w.devices), {'01:02:03:04:05:06', '02:03:04:05:06:07'})
            self.assertTrue(w.devices['01:02:03:04:05:06'].paired)
            self.assertFalse(w.devices['02:03:04:05:06:07'].paired)
        asyncio.run(run())

    def test_scan_missing_bluez_is_actionable(self):
        async def run():
            w = self.new_worker()
            with patch('mindflex.bluetooth_linux.shutil.which', return_value=None):
                await w.start_scan()
            messages = self.messages(w)
            self.assertEqual([item[0] for item in messages][-2:], ['scan_stopped', 'error'])
            self.assertIn('bluez', messages[-1][1].lower())
        asyncio.run(run())

    def test_scan_adapter_off_is_actionable(self):
        async def run():
            w = self.new_worker()
            w._call = AsyncMock(return_value=(0, 'Powered: no'))
            with patch('mindflex.bluetooth_linux.shutil.which', return_value='/usr/bin/bluetoothctl'):
                await w.start_scan()
            self.assertIn('powered off', self.messages(w)[-1][1])
        asyncio.run(run())

    def test_sdp_channel_order_and_fallback(self):
        async def run():
            w = self.new_worker()
            w._call = AsyncMock(return_value=(0, 'Service Name: Serial Port\n Channel: 9'))
            with patch('mindflex.bluetooth_linux.shutil.which', return_value='/usr/bin/sdptool'):
                channels = await w._channels('00:11:22:33:44:55')
            self.assertEqual(channels, [9, 1, 2, 3, 4, 5])
        asyncio.run(run())

    def test_successful_probe_confirms_before_forwarding_bytes(self):
        async def run():
            trace = []
            w = self.new_worker(on_connecting=lambda e: trace.append('connecting'),
                                on_connected=lambda e: trace.append('connected'),
                                on_bytes=lambda data: trace.append('bytes'),
                                on_disconnected=lambda: trace.append('disconnected'))
            address = '00:11:22:33:44:55'
            w.devices[address] = _Device(address, 'MindFlex', paired=True)
            a, b = socket.socketpair()
            a.setblocking(False)
            b.sendall(make_packet([0x80, 2, 0, 10]) * 5)
            w._channels = AsyncMock(return_value=[3])
            with patch.object(w, '_open_socket', return_value=a):
                await w.connect(address)
            self.assertEqual(trace[:3], ['connecting', 'connected', 'bytes'])
            self.assertEqual(w.current_endpoint.backend, 'bluez-rfcomm')
            self.assertEqual(len([e for e in self.messages(w) if e[0]=='mindflex_confirmed']), 1)
            await w.disconnect()
            b.close()
            self.assertEqual(trace.count('disconnected'), 1)
            self.assertIsNone(w.current_endpoint)
        asyncio.run(run())

    def test_invalid_thinkgear_bytes_cannot_reach_controller(self):
        async def run():
            trace = []
            w = self.new_worker(on_connected=lambda e: trace.append('connected'),
                                on_bytes=lambda data: trace.append('bytes'))
            address = '00:11:22:33:44:55'
            w.devices[address] = _Device(address, 'Test transmitter', paired=True)
            a, b = socket.socketpair()
            a.setblocking(False)
            b.sendall(b'noise' + make_packet([0x04, 80])*5)  # Valid checksums; NO RAW.
            b.shutdown(socket.SHUT_WR)
            w._channels = AsyncMock(return_value=[1])
            with patch.object(w, '_open_socket', return_value=a):
                await w.connect(address)
            self.assertEqual(trace, [])
            self.assertIn('No RFCOMM channel', self.messages(w)[-1][1])
            b.close()
        asyncio.run(run())

    def test_auto_pairing_failure_has_guidance(self):
        async def run():
            w = self.new_worker()
            address = '00:11:22:33:44:55'
            w.devices[address] = _Device(address, 'MindFlex', paired=False)
            w._call = AsyncMock(return_value=(1, 'Authentication rejected'))
            await w.connect(address)
            self.assertIn('Bluetooth Settings', self.messages(w)[-1][1])
            w._call.assert_awaited_once()
        asyncio.run(run())

    def test_broken_stream_retires_endpoint(self):
        async def run():
            trace = []
            w = self.new_worker(on_connected=lambda e: trace.append('connected'),
                                on_disconnected=lambda: trace.append('disconnected'))
            address = '00:11:22:33:44:55'
            w.devices[address] = _Device(address, 'MindFlex', paired=True)
            a, b = socket.socketpair()
            a.setblocking(False)
            b.sendall(make_packet([0x80, 2, 0, 10])*5)
            w._channels = AsyncMock(return_value=[1])
            with patch.object(w, '_open_socket', return_value=a):
                await w.connect(address)
            b.close()
            await asyncio.sleep(0.1)
            self.assertEqual(trace, ['connected', 'disconnected'])
            self.assertIsNone(w.current_endpoint)
        asyncio.run(run())

    def test_paired_device_stays_on_paired_path(self):
        async def run():
            w = self.new_worker()
            address='00:11:22:33:44:55'
            w.devices[address] = _Device(address,'MindFlex',True)
            w._channels = AsyncMock(return_value=[])
            w._call = AsyncMock(return_value=(1,'should not pair'))
            await w.connect(address)
            w._call.assert_not_awaited()
            self.assertIn('No RFCOMM channel', self.messages(w)[-1][1])
        asyncio.run(run())

    def test_controller_receives_valid_probe_raw_and_retires_on_disconnect(self):
        from mindflex.controller import EEGController
        controller = EEGController()
        async def run():
            w = self.new_worker(
                on_connecting=controller.prepare_bluetooth_connection,
                on_connected=controller.confirm_bluetooth_connection,
                on_bytes=controller.feed_bluetooth_bytes,
                on_disconnected=controller.bluetooth_disconnected,
                on_error=controller.bluetooth_error,
            )
            address = '00:11:22:33:44:55'
            w.devices[address] = _Device(address, 'MindFlex', paired=True)
            a, b = socket.socketpair()
            a.setblocking(False)
            packets = b''.join(make_packet([0x80, 2, 0, v]) for v in range(16))
            b.sendall(packets)
            w._channels = AsyncMock(return_value=[1])
            with patch.object(w, '_open_socket', return_value=a):
                await w.connect(address)
            await asyncio.sleep(0.08)  # Allow the controller's ingestion thread to flush.
            self.assertEqual(controller.snapshot().raw_total_samples, 16)
            self.assertTrue(controller.snapshot().connected)
            self.assertEqual(controller.raw_slice(0, 16), list(range(16)))
            await w.disconnect()
            self.assertFalse(controller.snapshot().connected)
            self.assertEqual(controller.snapshot().raw_total_samples, 0)
            b.close()
        try:
            asyncio.run(run())
        finally:
            controller.close()

    def test_failed_channel_falls_back_to_next_and_probes(self):
        async def run():
            w = self.new_worker()
            address = '00:11:22:33:44:55'
            w.devices[address] = _Device(address, 'MindFlex', paired=True)
            a, b = socket.socketpair()
            a.setblocking(False)
            b.sendall(make_packet([0x80, 2, 0, 1]) * 5)
            channels = []
            def open_channel(addr, channel, timeout):
                channels.append(channel)
                if channel == 1:
                    raise OSError('refused')
                return a
            w._channels = AsyncMock(return_value=[1, 3])
            with patch.object(w, '_open_socket', side_effect=open_channel):
                await w.connect(address)
            self.assertEqual(channels, [1, 3])
            self.assertIsNotNone(w.current_endpoint)
            await w.disconnect()
            b.close()
        asyncio.run(run())

    def test_unpaired_device_can_pair_through_bluez(self):
        async def run():
            w = self.new_worker()
            address = '00:11:22:33:44:55'
            w.devices[address] = _Device(address, 'MindFlex', paired=False)
            w._call = AsyncMock(return_value=(0, 'Pairing successful'))
            w._channels = AsyncMock(return_value=[])
            await w.connect(address)
            self.assertTrue(w.devices[address].paired)
            self.assertEqual(w._call.await_args.args[0][:3], ['bluetoothctl', '--timeout', '15'])
            self.assertEqual(w._call.await_args.args[0][-1], address)
        asyncio.run(run())

    def test_empty_scan_cannot_activate_bci(self):
        async def run():
            trace = []
            w = self.new_worker(on_connecting=lambda e: trace.append('connecting'))
            await w.connect('00:11:22:33:44:55')
            self.assertEqual(trace, [])
            self.assertIsNone(w.current_endpoint)
            self.assertEqual(self.messages(w)[-1][0], 'connect_failed')
        asyncio.run(run())


if __name__ == '__main__':
    unittest.main()
