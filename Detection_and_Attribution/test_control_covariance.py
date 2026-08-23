import unittest

import numpy as np

from control_covariance import estimate_control_covariance, shrink_covariance


class ControlCovarianceTests(unittest.TestCase):
    def test_stacks_times_within_each_trajectory(self):
        initial = np.array([-1.0, 0.0, 2.0])
        covariance, empirical, vectors = estimate_control_covariance(
            initial,
            drift=lambda x: np.ones_like(x),
            observables=(lambda x: x, lambda x: x**2),
            n_steps=2,
            dt=0.5,
            sigma=0.0,
            rng=np.random.default_rng(7),
            alpha=0.1,
        )

        expected_vectors = np.array(
            [
                [-0.5, 0.0, 0.25, 0.0],
                [0.5, 1.0, 0.25, 1.0],
                [2.5, 3.0, 6.25, 9.0],
            ]
        )
        np.testing.assert_allclose(vectors, expected_vectors)
        np.testing.assert_allclose(empirical, np.cov(expected_vectors, rowvar=False))
        np.testing.assert_allclose(covariance, shrink_covariance(empirical, 0.1))
        np.testing.assert_allclose(covariance, covariance.T)
        self.assertGreater(np.linalg.eigvalsh(covariance)[0], 0.0)

    def test_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            estimate_control_covariance(
                np.array([0.0]),
                drift=lambda x: x,
                observables=(lambda x: x,),
                n_steps=1,
                dt=0.1,
                sigma=0.0,
                rng=np.random.default_rng(1),
            )
        with self.assertRaises(ValueError):
            shrink_covariance(np.eye(2), alpha=1.1)


if __name__ == "__main__":
    unittest.main()
