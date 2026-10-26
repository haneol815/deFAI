import numpy as np
import pytest

from defai.aggregation import (
    coordinate_median,
    krum_scores,
    multi_krum_select,
    pairwise_sq_distances,
    trimmed_mean,
    weighted_mean,
)


def U(*values, dtype=np.float32):
    return {"w": np.array(values, dtype=dtype)}


def test_weighted_mean_matches_manual_and_keeps_dtype():
    out = weighted_mean([U(1, 2), U(3, 6)], [1, 3])
    np.testing.assert_allclose(out["w"], [2.5, 5.0])
    assert out["w"].dtype == np.float32


def test_weighted_mean_rejects_bad_weights():
    with pytest.raises(ValueError):
        weighted_mean([U(1), U(2)], [0, 0])
    with pytest.raises(ValueError):
        weighted_mean([U(1), U(2)], [1])


def test_median_ignores_outliers():
    updates = [U(1, 1), U(2, 2), U(3, 3), U(1000, -1000), U(2, 2)]
    np.testing.assert_allclose(coordinate_median(updates)["w"], [2, 2])


def test_trimmed_mean_drops_extremes_per_coordinate():
    updates = [U(0, 10), U(1, 11), U(2, 12), U(3, 13), U(100, -100)]
    # n=5, ratio=0.2 -> drop 1 from each side per coordinate
    out = trimmed_mean(updates, 0.2)["w"]
    np.testing.assert_allclose(out, [2.0, 11.0])


def test_trimmed_mean_zero_ratio_is_mean_and_bad_ratio_rejected():
    updates = [U(0), U(2), U(10)]
    np.testing.assert_allclose(trimmed_mean(updates, 0.0)["w"], [4.0])
    with pytest.raises(ValueError):
        trimmed_mean(updates, 0.5)


def test_pairwise_distances_symmetric_zero_diagonal():
    d = pairwise_sq_distances([U(0, 0), U(3, 4), U(6, 8)])
    np.testing.assert_allclose(d, [[0, 25, 100], [25, 0, 25], [100, 25, 0]])


def test_krum_scores_hand_computed():
    # scalars 0, 1, 10 with f=0: n-f-2 = 1 nearest neighbour each
    d = pairwise_sq_distances([U(0), U(1), U(10)])
    np.testing.assert_allclose(krum_scores(d, 0), [1, 1, 81])


def _honest_and_byzantine(n_honest=7, n_byz=2, dim=50, seed=0):
    rng = np.random.default_rng(seed)
    honest = [{"w": rng.normal(0, 0.1, dim)} for _ in range(n_honest)]
    byz = [{"w": rng.normal(50, 1, dim)} for _ in range(n_byz)]
    return honest + byz


def test_krum_and_multikrum_pick_honest_clients():
    updates = _honest_and_byzantine()  # n=9 >= 2f+3 with f=2
    assert multi_krum_select(updates, num_byzantine=2, num_select=1)[0] < 7
    chosen = multi_krum_select(updates, num_byzantine=2, num_select=5)
    assert len(chosen) == 5 and all(i < 7 for i in chosen)


def test_krum_single_client_and_oversized_select():
    assert multi_krum_select([U(1)], 0, 3) == [0]
    assert sorted(multi_krum_select([U(0), U(1), U(2)], 0, 10)) == [0, 1, 2]
