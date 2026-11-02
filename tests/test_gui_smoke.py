"""Real Tk widget lifecycle smoke test; run under Xvfb on Linux CI."""
from __future__ import annotations
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from mindflex.settings import Settings


@unittest.skipUnless(os.environ.get('MIND_FLEX_RUN_GUI_TESTS') == '1',
                     'GUI smoke requires a graphical display (xvfb-run on Linux)')
class GUISmokeTests(unittest.TestCase):
    def test_app_creation_simulator_navigation_and_clean_shutdown(self):
        from mindflex.ui import MindFlexApp
        from mindflex.user_profile import UserProfile
        from mindflex.rules import WorkflowStep
        with tempfile.TemporaryDirectory() as td:
            profile = UserProfile('Simulated Test User', 'simulated-test', Path(td)/'profile')
            with patch.object(MindFlexApp,'_select_user_profile',return_value=profile), \
                 patch.object(Settings,'load',return_value=Settings(connection_mode='serial')), \
                 patch.object(Settings,'save',return_value=None):
                app=MindFlexApp()
                try:
                    app.current_view.start_simulator()
                    deadline=time.monotonic()+2.0
                    while app.controller.snapshot().raw_total_samples < 128 and time.monotonic()<deadline:
                        app.update();time.sleep(.02)
                    snap=app.controller.snapshot()
                    self.assertTrue(snap.connected)
                    self.assertGreaterEqual(snap.raw_total_samples,128)
                    for step in (WorkflowStep.MONITOR, WorkflowStep.NEURO_CONTROL,
                                 WorkflowStep.DIAGNOSTICS, WorkflowStep.CONNECTION):
                        app.navigate(step)
                        app.update()
                        self.assertEqual(app.current_step,step)
                finally:
                    app._close()

if __name__=='__main__': unittest.main()
