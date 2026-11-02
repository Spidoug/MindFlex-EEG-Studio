"""Transport framing and sample-grid regressions without a device."""
import unittest

from mindflex.bci import bci_window, latest_bci_window
from mindflex.parser import ThinkGearParser, make_packet
from mindflex.settings import BCI_EPOCH_SAMPLES, BCI_STEP_SAMPLES


class FramingAndTimingTests(unittest.TestCase):
    def test_fragmented_raw_and_metrics(self):
        payload = [0x02, 0, 0x04, 77, 0x05, 33, 0x80, 2, 0xFF, 0x9C]
        packet = make_packet(payload)
        parser = ThinkGearParser()
        outputs = []
        for byte in packet:
            outputs += parser.feed(bytes([byte]))
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].raw, [-100])
        self.assertEqual((outputs[0].attention, outputs[0].meditation), (77, 33))
        self.assertEqual(parser.bad_checksums, 0)

    def test_bad_checksum_recovery(self):
        packet = make_packet([0x80, 2, 0, 42])
        bad = packet[:-1] + bytes([packet[-1] ^ 0x40])
        parser = ThinkGearParser()
        result = parser.feed(bad + packet)
        self.assertEqual(parser.bad_checksums, 1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].raw, [42])

    def test_fixed_grid(self):
        self.assertIsNone(latest_bci_window(BCI_EPOCH_SAMPLES - 1))
        self.assertEqual(latest_bci_window(BCI_EPOCH_SAMPLES), bci_window(0, 0))
        self.assertEqual(latest_bci_window(BCI_EPOCH_SAMPLES + 2 * BCI_STEP_SAMPLES), bci_window(0, 2))
        self.assertEqual(bci_window(80, 2).start_sample, 80 + 2 * BCI_STEP_SAMPLES)

if __name__ == '__main__':
    unittest.main()
