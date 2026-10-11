import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from report import MATCH_ARGS, load_matched_runs, paired_summary


class TestMatchedReport(unittest.TestCase):
    def make_run(self, root, seed, lr=.001):
        path = root / str(seed)
        path.mkdir()
        args = {key: 7 for key in MATCH_ARGS}
        args.update(seed=seed, lr=lr, methods=['rate_adapter', 'blind_adapter'])
        subjects = ['A', 'B', 'C', 'D']
        protocol = {'arguments': args, 'subjects': {'test': subjects},
                    'dataset_provenance': {'files': ['fixed']}, 'checkpoint_sha256': 'fixed',
                    'experiment_source_sha256': {'run.py': 'reporting', 'model.py': 'fixed'}}
        (path / 'protocol.json').write_text(json.dumps(protocol))
        records, subject_records = [], []
        for method in args['methods']:
            gain = .05 if method == 'rate_adapter' else 0.
            for condition in ('clean30', 'aa15', 'aa10', 'unseen12', 'unseen6', 'naive15', 'naive10'):
                record = {'method': method, 'condition': condition, 'accuracy': .5 + gain,
                          'macro_f1': .3 + gain, 'nll': 1. - gain}
                records.append(record)
                subject_records.extend({**record, 'subject': s} for s in subjects)
        pd.DataFrame(records).to_csv(path / 'metrics.csv', index=False)
        pd.DataFrame(subject_records).to_csv(path / 'per_subject.csv', index=False)
        return path

    def test_pairing_and_seeds_are_not_pooled_as_independent_subjects(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = load_matched_runs([self.make_run(root, s) for s in (7, 17, 27)], 3)
            result = paired_summary(pd.concat([r[2] for r in runs]), pd.concat([r[3] for r in runs]), 50, 7)
            self.assertEqual(len(result), 21)
            self.assertTrue((result.seeds == 3).all())
            self.assertTrue((result.subjects == 4).all())
            self.assertTrue((result.positive_seeds == 3).all())
            self.assertTrue((result.subject_bootstrap_lo > .049).all())

    def test_reject_duplicate_seeds_and_mismatched_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = self.make_run(root, 7)
            b = self.make_run(root, 17, lr=.01)
            with self.assertRaisesRegex(ValueError, 'Repeated model seed'):
                load_matched_runs([a, a], 2)
            with self.assertRaisesRegex(ValueError, 'differ'):
                load_matched_runs([a, b], 2)
            with self.assertRaisesRegex(ValueError, 'at least 3'):
                load_matched_runs([a], 3)

    def test_reject_incomplete_run(self):
        with tempfile.TemporaryDirectory() as directory:
            a = self.make_run(Path(directory), 7)
            metrics = pd.read_csv(a / 'metrics.csv')
            metrics.iloc[:-1].to_csv(a / 'metrics.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                load_matched_runs([a], 1)


if __name__ == '__main__':
    unittest.main()
