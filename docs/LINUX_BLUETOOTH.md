# Bluetooth Classic on Linux (BlueZ)

The Linux Studio uses the **BlueZ Bluetooth Classic / BR/EDR** stack. It does not import Windows WinRT or use Bluetooth Low Energy APIs such as Bleak. The MindFlex/TGAM serial stream is opened through a Linux RFCOMM socket and validated with ThinkGear RAW packets before the EEG/BCI controller receives bytes.

## Ubuntu / Debian installation

```bash
sudo apt update
sudo apt install bluez python3-tk python3-venv python3-pip
sudo systemctl enable --now bluetooth
bluetoothctl show
```

`Powered: yes` is required. If powered off, use your distribution's Bluetooth settings or run `bluetoothctl power on`. The script does not silently acquire administrator privileges. A kernel and Python build with `AF_BLUETOOTH`/`BTPROTO_RFCOMM` support is required; this is present in typical distribution Python builds. The application runs without `sudo`.

## Discovery, pairing, and first connection

1. Turn on your Bluetooth adapter and MindFlex transmitter. Pair the MindFlex through your desktop Bluetooth settings. If the device requests a PIN, use the value documented for **your adapter**, not an assumed universal code.
2. Run `./RUN_STUDIO.sh`, select **Bluetooth**, and choose **Search**. BlueZ scans for BR/EDR devices and lists discovered and already paired devices.
3. Select your headset and press **Connect**. The worker optionally attempts pairing of unpaired devices; PIN prompts are not handled inside the Studio, so system pairing may be necessary first.
4. The worker retrieves the SPP channel using `sdptool browse` if installed, then tries a bounded set of common RFCOMM channels when SDP is unavailable. **Only a channel that returns valid ThinkGear RAW packets is accepted.** Failed probes never reach EEG processing.
5. A disconnect discards the stream and the BCI controller's prior sample continuity. Use Search / Connect to reconnect.

`sdptool` is optional (usually part of BlueZ utilities), but strongly recommended for adapters whose SPP channel is not 1–5. Bluetooth Classic SPP is not supported by generic BLE-only adapters or APIs.

For manual Bluetooth troubleshooting:

```bash
bluetoothctl show
bluetoothctl devices
bluetoothctl devices Paired
bluetoothctl scan bredr
# inside bluetoothctl, after discovering the headset:
# pair AA:BB:CC:DD:EE:FF
# trust AA:BB:CC:DD:EE:FF
# quit
```

On some systems a graphical Bluetooth manager handles pairing prompts better than the non-interactive `bluetoothctl` command.

## Troubleshooting

| Symptom | What to verify |
| --- | --- |
| BlueZ not found | Install `bluez` and check that `bluetoothctl` is on PATH |
| Adapter off / absent | `bluetoothctl show`, `rfkill list`, desktop Bluetooth controls, service status |
| No device | Device powered and discoverable, `bluetoothctl scan bredr`, previously paired devices |
| Pairing fails | Pair using the desktop Bluetooth panel, with the adapter-specific PIN if asked |
| Connection refused / timeouts | SPP service active, `sdptool browse <MAC>`, adapter already connected in another app |
| RFCOMM opens but probe fails | MindFlex must output **ThinkGear RAW** at the supported baud / mode; connecting to an unrelated RFCOMM service does not count |
| Python lacks RFCOMM sockets | Use Linux distribution CPython rather than a custom build without Bluetooth socket support |

`./RUN_STUDIO.sh --diagnose` reports Python Bluetooth socket support and `bluetoothctl` availability without requiring an adapter, and logs errors in `logs/studio-startup.log`. The Simulator and Serial modes remain functional even without Bluetooth hardware.

## Test scope

The automated tests emulate BlueZ command output, RFCOMM stream reads, valid/invalid ThinkGear probes, cleanup, and the application's callback sequence. Hardware pairing, adapter driver differences, RF interference, service discovery, sustained EEG timing and manual PIN exchange **must still be tested with a real MindFlex headset on the target Linux machine**.
