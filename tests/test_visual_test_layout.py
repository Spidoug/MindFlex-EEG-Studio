"""GUI regression tests for the cue-first, bilingual test workspace.

Run with MIND_FLEX_RUN_GUI_TESTS=1 under Xvfb on Linux or a Windows desktop.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mindflex.neural_visual import COMMAND_IDS
from mindflex.settings import Settings


@unittest.skipUnless(os.environ.get("MIND_FLEX_RUN_GUI_TESTS") == "1", "A Tk display is required")
class VisualWorkspaceTests(unittest.TestCase):
    def test_layout_at_compact_and_wide_sizes_and_focus_lifecycle(self):
        from mindflex.ui import MindFlexApp
        from mindflex.rules import NeuroStep, WorkflowStep
        from mindflex.user_profile import UserProfile

        with tempfile.TemporaryDirectory() as directory:
            user = UserProfile("UI Test User", "ui-test", Path(directory) / "profile")
            with (patch.object(MindFlexApp, "_select_user_profile", return_value=user),
                  patch.object(Settings, "load", return_value=Settings(connection_mode="serial")),
                  patch.object(Settings, "save", return_value=None)):
                app = MindFlexApp()
                try:
                    app.current_view.start_simulator()
                    app.update()
                    app.navigate(WorkflowStep.NEURO_CONTROL)
                    view = app.current_view
                    view.stage = NeuroStep.LIVE_CONTROL
                    view._render_stage()
                    app.update()
                    self.assertFalse(view.test_split.secondary_visible)
                    self.assertEqual(str(view.test_start_button.cget("state")), "normal")
                    self.assertEqual(str(view.test_stop_button.cget("state")), "disabled")
                    for label in COMMAND_IDS:
                        view._draw_command_symbol(view.test_canvas, label, "")
                        self.assertEqual(len(view.test_canvas.find_withtag("cue-symbol")), 1)
                        bounds = view.test_canvas.bbox("cue-symbol")
                        self.assertIsNotNone(bounds)
                        self.assertGreater(bounds[2] - bounds[0], 20)
                    view.test_show_details_var.set(True)
                    view._toggle_test_details()
                    app.update()
                    self.assertTrue(view.test_split.secondary_visible)
                    view.test_show_details_var.set(False)
                    view._toggle_test_details()
                    app.update()
                    self.assertFalse(view.test_split.secondary_visible)
                    view._open_test_focus()
                    app.update()
                    self.assertTrue(view._test_focus_window.winfo_exists())
                    self.assertEqual(len(view._test_focus_canvas.find_withtag("cue-symbol")), 1)
                    # A second invocation must focus the existing window, not add another.
                    window = view._test_focus_window
                    view._open_test_focus()
                    self.assertIs(view._test_focus_window, window)
                    window.event_generate("<Escape>")
                    app.update()
                    self.assertIsNone(view._test_focus_window)
                    for size in ("1024x680", "1440x900"):
                        app.geometry(size)
                        app.update()
                        view._refresh_command_test_visual()
                        self.assertGreater(view.test_canvas.winfo_width(), 300)
                        self.assertGreater(view.test_canvas.winfo_height(), 180)
                        self.assertEqual(len(view.test_canvas.find_withtag("cue-symbol")), 1)
                finally:
                    app._close()


if __name__ == "__main__":
    unittest.main()
