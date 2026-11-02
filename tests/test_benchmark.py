"""Benchmark is also a regression gate in offline CI."""
import unittest
from mindflex.benchmark import run


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = run(performance_epochs=4)

    def test_all_offline_scenarios(self):
        self.assertEqual(self.report['fail_count'], 0, self.report['scenarios'])
        self.assertEqual(self.report['pass_count'], 19)

    def test_decision_accounting_and_disclaimer(self):
        item = next(s['metrics'] for s in self.report['scenarios'] if s['name'] == 'training_prediction_stabilization')
        self.assertEqual(item['synthetic_decisions'] + item['abstentions'], 24)
        self.assertIn('NOT human', item['caveat'])

    def test_population_dropout_is_rejected(self):
        metrics = next(x['metrics'] for x in self.report['scenarios'] if x['name'] == 'population_realistic_sessions')
        self.assertEqual(metrics['results']['dropout']['rejected'], 32)
        self.assertEqual(metrics['results']['none']['rejected'], 0)

    def test_cross_profile_limit_is_reported(self):
        metrics = next(x['metrics'] for x in self.report['scenarios'] if x['name'] == 'cross_session_domain_shift')
        self.assertEqual(len(metrics['domain_shift']), 4)
        self.assertTrue(any(p['conditional_accuracy'] < 0.75 for p in metrics['domain_shift'].values()))

    def test_recording_integrity_scenario(self):
        item = next(s['metrics'] for s in self.report['scenarios']
                    if s['name'] == 'controller_record_replay_integrity')
        self.assertEqual(item['integrity_boundary_sample'], 384)
        self.assertTrue(item['retired_on_disconnect'])

    def test_heldout_muscle_detection(self):
        item = next(s['metrics'] for s in self.report['scenarios']
                    if s['name'] == 'heldout_artifact_guard')
        self.assertGreaterEqual(item['holdout_conditions']['emg']['rejected'], 58)

    def test_real_time_rate_is_explicit(self):
        item = next(s['metrics'] for s in self.report['scenarios'] if s['name'] == 'feature_throughput')
        self.assertEqual(item['live_required_epochs_per_second'], 4)


if __name__ == '__main__':
    unittest.main()
