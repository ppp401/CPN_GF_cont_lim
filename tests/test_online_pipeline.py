import tempfile
import unittest
from pathlib import Path

import numpy as np

from cpn_gf.analysis import analyze_run
from cpn_gf.io import atomic_json, atomic_npz


class OnlineAnalysisTests(unittest.TestCase):
    def test_analysis_reads_chunked_observables(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rho = [0.05, 0.1]
            manifest = {"flow": {"rho": rho}, "L": 8, "chains": 3,
                        "sampling": {"relative_error": 1.0}}
            atomic_json(root / "manifest.json", manifest)
            atomic_npz(root / "scale.npz", xi=np.asarray(2.0),
                       xi_loo=np.asarray([2.0, 2.0, 2.0]))
            rng = np.random.default_rng(2)
            shape = (3, 4, 4)
            values = np.stack((rng.random(shape) + 1.0,
                               rng.random(shape) + 4.0,
                               rng.random(shape) + 1.0,
                               rng.random(shape) + 1.0,
                               rng.integers(-2, 3, shape),
                               rng.integers(-2, 3, shape)), axis=-1)
            q_s = rng.integers(-3, 4, (3, 4))
            atomic_npz(root / "observations" / "flow_00000000.npz", values=values, Q_s=q_s,
                       output_steps=np.asarray([0, 10, 20, 40]),
                       times=np.asarray([0.0, 0.1, 0.2, 0.4]))
            result, summary = analyze_run(root)
            self.assertEqual(result["tE_action"].shape, (2,))
            self.assertEqual(result["n_samples_per_chain"].tolist(), [4, 4, 4])
            self.assertTrue(np.isfinite(summary["maximum_tE_relative_error"]))
            self.assertIn("unflowed_chi_t_Q_U", result)
            self.assertTrue(bool(result["unflowed_Q_s_applicable"]))


if __name__ == "__main__":
    unittest.main()
