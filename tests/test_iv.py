"""Tests for IV and PSI computation functions."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest


# ---------------------------------------------------------------------------
# Inline implementations (no data dependency)
# ---------------------------------------------------------------------------
def iv_standard_error(
    bin_good: npt.ArrayLike,
    bin_bad: npt.ArrayLike,
    total_good: int,
    total_bad: int,
) -> tuple[float, float]:
    bg = np.asarray(bin_good, dtype=float)
    bb = np.asarray(bin_bad, dtype=float)
    p_g = bg / total_good
    p_b = bb / total_bad
    woe = np.log(np.clip(p_b, 1e-15, None) / np.clip(p_g, 1e-15, None))
    weight = p_b - p_g
    iv = float((weight * woe).sum())
    se_woe = np.sqrt(1.0 / np.clip(bg, 1, None) + 1.0 / np.clip(bb, 1, None))
    se_iv = float(np.sqrt((weight**2 * se_woe**2).sum()))
    return iv, se_iv


def compute_psi_with_se(
    ref_counts: npt.ArrayLike,
    comp_counts: npt.ArrayLike,
) -> tuple[float, float]:
    ref = np.asarray(ref_counts, dtype=float)
    comp = np.asarray(comp_counts, dtype=float)
    p = np.clip(ref / ref.sum(), 1e-15, None)
    q = np.clip(comp / comp.sum(), 1e-15, None)
    ln_ratio = np.log(q / p)
    psi = float(((q - p) * ln_ratio).sum())
    se_ln = np.sqrt(1.0 / np.clip(ref, 1, None) + 1.0 / np.clip(comp, 1, None))
    se_psi = float(np.sqrt(((q - p) ** 2 * se_ln**2).sum()))
    return psi, se_psi


# ---------------------------------------------------------------------------
# IV tests
# ---------------------------------------------------------------------------
class TestIVStandardError:
    def test_identical_distributions_give_zero_iv(self) -> None:
        """When good and bad have the same distribution, IV = 0."""
        bin_good = np.array([100, 100, 100])
        bin_bad = np.array([50, 50, 50])
        iv, se = iv_standard_error(bin_good, bin_bad, 300, 150)
        assert iv == pytest.approx(0.0, abs=1e-10)

    def test_iv_is_nonnegative(self) -> None:
        """IV (Jeffreys divergence) is always >= 0."""
        rng = np.random.default_rng(42)
        for _ in range(50):
            n_bins = rng.integers(2, 10)
            bin_good = rng.integers(10, 500, size=n_bins)
            bin_bad = rng.integers(10, 500, size=n_bins)
            iv, se = iv_standard_error(bin_good, bin_bad, int(bin_good.sum()), int(bin_bad.sum()))
            assert iv >= -1e-10, f"IV should be non-negative, got {iv}"
            assert se >= 0, f"SE should be non-negative, got {se}"

    def test_se_decreases_with_sample_size(self) -> None:
        """Doubling sample size should reduce SE."""
        bg_small = np.array([30, 20, 50])
        bb_small = np.array([10, 25, 15])
        _, se_small = iv_standard_error(bg_small, bb_small, 100, 50)

        bg_large = bg_small * 10
        bb_large = bb_small * 10
        _, se_large = iv_standard_error(bg_large, bb_large, 1000, 500)

        assert se_large < se_small

    def test_known_two_bin_case(self) -> None:
        """Verify IV against manual calculation for a simple 2-bin case."""
        # bin 0: 40 good, 10 bad;  bin 1: 60 good, 90 bad
        bg = np.array([40, 60])
        bb = np.array([10, 90])
        total_g, total_b = 100, 100

        p_g = bg / total_g  # [0.4, 0.6]
        p_b = bb / total_b  # [0.1, 0.9]
        woe = np.log(p_b / p_g)
        expected_iv = float(((p_b - p_g) * woe).sum())

        iv, se = iv_standard_error(bg, bb, total_g, total_b)
        assert iv == pytest.approx(expected_iv, rel=1e-10)
        assert se > 0

    def test_se_formula_matches_manual(self) -> None:
        """Check SE formula against hand computation."""
        bg = np.array([200, 300])
        bb = np.array([50, 150])
        total_g, total_b = 500, 200

        p_g = bg / total_g
        p_b = bb / total_b
        weight = p_b - p_g
        se_woe = np.sqrt(1.0 / bg + 1.0 / bb)
        expected_se = float(np.sqrt((weight**2 * se_woe**2).sum()))

        _, se = iv_standard_error(bg, bb, total_g, total_b)
        assert se == pytest.approx(expected_se, rel=1e-10)


# ---------------------------------------------------------------------------
# PSI tests
# ---------------------------------------------------------------------------
class TestPSI:
    def test_identical_distributions_give_zero_psi(self) -> None:
        """PSI between identical distributions is 0."""
        counts = np.array([100, 200, 150, 50])
        psi, se = compute_psi_with_se(counts, counts)
        assert psi == pytest.approx(0.0, abs=1e-10)

    def test_psi_is_nonnegative(self) -> None:
        """PSI (Jeffreys divergence) is always >= 0."""
        rng = np.random.default_rng(99)
        for _ in range(50):
            n_bins = rng.integers(2, 10)
            ref = rng.integers(10, 500, size=n_bins)
            comp = rng.integers(10, 500, size=n_bins)
            psi, se = compute_psi_with_se(ref, comp)
            assert psi >= -1e-10
            assert se >= 0

    def test_psi_is_symmetric(self) -> None:
        """PSI(P, Q) == PSI(Q, P) — it's a symmetric divergence."""
        ref = np.array([100, 200, 300])
        comp = np.array([150, 150, 300])
        psi_fwd, _ = compute_psi_with_se(ref, comp)
        psi_bwd, _ = compute_psi_with_se(comp, ref)
        assert psi_fwd == pytest.approx(psi_bwd, rel=1e-10)

    def test_se_decreases_with_sample_size(self) -> None:
        """Scaling counts up should reduce SE."""
        ref = np.array([50, 100, 50])
        comp = np.array([80, 70, 50])
        _, se_small = compute_psi_with_se(ref, comp)
        _, se_large = compute_psi_with_se(ref * 10, comp * 10)
        assert se_large < se_small


# ---------------------------------------------------------------------------
# Location invariance (Welford property)
# ---------------------------------------------------------------------------
class TestLocationInvariance:
    def test_variance_invariant_to_constant_shift(self) -> None:
        """Var(X - K) == Var(X) for any constant K."""
        rng = np.random.default_rng(42)
        p, n = 0.6, 100
        n_sim = 5_000

        log_odds: list[float] = []
        woe: list[float] = []
        K = np.log(0.4 / 0.6)  # prior log odds (constant)

        for _ in range(n_sim):
            nb = rng.binomial(n, p)
            ng = n - nb
            if nb > 0 and ng > 0:
                theta = np.log(nb / ng)
                log_odds.append(theta)
                woe.append(theta - K)

        var_lo = np.var(log_odds, ddof=1)
        var_woe = np.var(woe, ddof=1)
        assert var_lo == pytest.approx(var_woe, rel=1e-6)
