"""
Tests for tg_purge/statistics.py — statistical sampling framework.

Uses pure-math functions only; no Telethon or network access.
All test users are represented as (user, score, reasons) tuples
matching the output format of scoring.score_user().
"""

import math
import pytest

from tg_purge.statistics import (
    wilson_score_interval,
    estimate_bot_rate,
    sample_quality_report,
    format_stats_summary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scored(n_flagged, n_total, threshold=3):
    """
    Build a minimal list of (user, score, reasons) tuples.

    n_flagged users get score=threshold (just at the boundary).
    Remaining users get score=0.
    'user' is just an integer ID — statistics.py must not inspect it.
    """
    scored = []
    for i in range(n_flagged):
        scored.append((i, threshold, ["some_signal"]))
    for i in range(n_flagged, n_total):
        scored.append((i, 0, []))
    return scored


# ---------------------------------------------------------------------------
# TestWilsonScoreInterval
# ---------------------------------------------------------------------------

class TestWilsonScoreInterval:
    """Tests for wilson_score_interval(successes, total, z=1.96)."""

    def test_50pct_rate_tight_ci_large_sample(self):
        """Large sample at 50% should give a tight CI around 0.5."""
        lower, upper = wilson_score_interval(500, 1000)
        # At n=1000, 95% CI for p=0.5 is roughly ±0.031
        assert abs((lower + upper) / 2 - 0.5) < 0.01, "midpoint should be near 0.5"
        assert (upper - lower) < 0.07, "CI width should be narrow for large n"
        assert lower > 0.46
        assert upper < 0.54

    def test_50pct_rate_wider_ci_small_sample(self):
        """Small sample at 50% should give a wider CI than large sample."""
        lower_small, upper_small = wilson_score_interval(5, 10)
        lower_large, upper_large = wilson_score_interval(500, 1000)
        width_small = upper_small - lower_small
        width_large = upper_large - lower_large
        assert width_small > width_large, "small sample must have wider CI"

    def test_zero_successes_floors_lower_at_zero(self):
        """Zero successes: lower bound must be exactly 0.0."""
        lower, upper = wilson_score_interval(0, 100)
        assert lower == 0.0
        assert upper > 0.0  # upper should still be positive
        assert upper < 0.1   # but small

    def test_all_successes_caps_upper_at_one(self):
        """All successes: upper bound must be exactly 1.0."""
        lower, upper = wilson_score_interval(100, 100)
        assert upper == 1.0
        assert lower < 1.0
        assert lower > 0.9  # lower should be high

    def test_zero_total_returns_zero_tuple(self):
        """Edge case: total=0 must return (0.0, 0.0) without division by zero."""
        lower, upper = wilson_score_interval(0, 0)
        assert lower == 0.0
        assert upper == 0.0

    def test_returns_floats_in_unit_interval(self):
        """Result values must always be in [0, 1]."""
        for s, n in [(1, 10), (5, 10), (9, 10), (50, 200)]:
            lower, upper = wilson_score_interval(s, n)
            assert 0.0 <= lower <= 1.0, f"lower={lower} out of range for s={s}, n={n}"
            assert 0.0 <= upper <= 1.0, f"upper={upper} out of range for s={s}, n={n}"
            assert lower <= upper, f"lower > upper for s={s}, n={n}"

    def test_known_values_approx(self):
        """
        Cross-check against well-known Wilson CI value.

        For p_hat=0.3, n=100, z=1.96 the Wilson lower ≈ 0.214, upper ≈ 0.400.
        Using the published formula:
          centre = (p_hat + z²/2n) / (1 + z²/n)
          half   = z * sqrt(p_hat*(1-p_hat)/n + z²/(4n²)) / (1 + z²/n)
        """
        lower, upper = wilson_score_interval(30, 100)
        assert 0.21 < lower < 0.23, f"lower={lower}"
        assert 0.39 < upper < 0.41, f"upper={upper}"


# ---------------------------------------------------------------------------
# TestEstimateBotRate
# ---------------------------------------------------------------------------

class TestEstimateBotRate:
    """Tests for estimate_bot_rate(scored_users, total_subscribers, threshold=3)."""

    def test_basic_estimation_30_of_100(self):
        """30 flagged out of 100 sampled, 1000 total subscribers."""
        scored = _make_scored(30, 100, threshold=3)
        result = estimate_bot_rate(scored, total_subscribers=1000, threshold=3)

        assert result["sample_size"] == 100
        assert result["flagged_count"] == 30
        assert result["total_subscribers"] == 1000

        # Point estimate should be near 0.30
        assert abs(result["point_estimate"] - 0.30) < 0.01

        # CI bounds should be reasonable
        assert 0.0 < result["ci_lower"] < result["point_estimate"]
        assert result["point_estimate"] < result["ci_upper"] <= 1.0

        # Margin of error is half the CI width
        expected_moe = (result["ci_upper"] - result["ci_lower"]) / 2
        assert abs(result["margin_of_error"] - expected_moe) < 1e-9

    def test_threshold_filtering(self):
        """Only users at or above threshold should be flagged."""
        scored = [
            (1, 4, ["a"]),  # above threshold=3
            (2, 3, ["b"]),  # equal to threshold — flagged
            (3, 2, ["c"]),  # below threshold — not flagged
            (4, 0, []),
        ]
        result = estimate_bot_rate(scored, total_subscribers=100, threshold=3)
        assert result["flagged_count"] == 2  # ids 1 and 2

    def test_empty_sample_returns_all_zeros(self):
        """Empty scored_users must not raise and must return zero dict."""
        result = estimate_bot_rate([], total_subscribers=500, threshold=3)
        assert result["sample_size"] == 0
        assert result["flagged_count"] == 0
        assert result["point_estimate"] == 0.0
        assert result["ci_lower"] == 0.0
        assert result["ci_upper"] == 0.0
        assert result["margin_of_error"] == 0.0
        assert result["total_subscribers"] == 500

    def test_all_flagged(self):
        """All users above threshold: upper CI should be capped at 1.0."""
        scored = _make_scored(50, 50, threshold=3)
        result = estimate_bot_rate(scored, total_subscribers=200, threshold=3)
        assert result["flagged_count"] == 50
        assert result["ci_upper"] == 1.0

    def test_none_flagged(self):
        """Zero flagged: lower CI should be 0.0."""
        scored = _make_scored(0, 50, threshold=3)
        result = estimate_bot_rate(scored, total_subscribers=200, threshold=3)
        assert result["flagged_count"] == 0
        assert result["ci_lower"] == 0.0

    def test_result_contains_all_required_keys(self):
        """Return dict must contain all seven documented keys."""
        scored = _make_scored(10, 50, threshold=3)
        result = estimate_bot_rate(scored, total_subscribers=500)
        required_keys = {
            "point_estimate", "ci_lower", "ci_upper", "margin_of_error",
            "sample_size", "flagged_count", "total_subscribers",
        }
        assert required_keys.issubset(result.keys())


# ---------------------------------------------------------------------------
# TestSampleQualityReport
# ---------------------------------------------------------------------------

class TestSampleQualityReport:
    """Tests for sample_quality_report(enumerated, total, query_stats)."""

    def test_full_coverage(self):
        """When enumerated == total, coverage_pct should be 100.0."""
        query_stats = [
            ("a", 50, 50),
            ("b", 50, 50),
        ]
        result = sample_quality_report(enumerated=100, total=100, query_stats=query_stats)
        assert result["coverage_pct"] == pytest.approx(100.0)

    def test_low_coverage(self):
        """When enumerated is 50% of total, coverage_pct should be ~50."""
        query_stats = [("a", 50, 50)]
        result = sample_quality_report(enumerated=50, total=100, query_stats=query_stats)
        assert result["coverage_pct"] == pytest.approx(50.0)

    def test_zero_total_returns_unknown(self):
        """Zero total must return coverage=0.0 and bias='unknown' without errors."""
        result = sample_quality_report(enumerated=0, total=0, query_stats=[])
        assert result["coverage_pct"] == 0.0
        assert result["estimated_bias"] == "unknown"

    def test_high_bias_detection(self):
        """
        Top 20% of queries producing >80% of results → 'high' bias.

        With 10 queries where the first 2 (top 20%) produce 90 out of 100 new results.
        """
        # 2 out of 10 queries produce 90/100 new results = 90% => high
        query_stats = [("q1", 50, 50), ("q2", 40, 40)]  # top 2: 90 new
        for i in range(8):
            query_stats.append((f"q{i+3}", 2, 1))  # remaining 8: 8 new total
        result = sample_quality_report(
            enumerated=98, total=500, query_stats=query_stats
        )
        assert result["estimated_bias"] == "high"

    def test_moderate_bias_detection(self):
        """
        Top 20% of queries producing >60% but ≤80% of results → 'moderate' bias.

        With 10 queries where the top 2 produce 70/100 new results.
        """
        query_stats = [("q1", 40, 40), ("q2", 30, 30)]  # top 2: 70 new
        for i in range(8):
            query_stats.append((f"q{i+3}", 5, 4))  # remaining 8: 30 new total (approx)
        # Recalculate: 8 * 4 = 32 new from remainder, top 70 / 102 total = 68.6% > 60%
        result = sample_quality_report(
            enumerated=102, total=500, query_stats=query_stats
        )
        assert result["estimated_bias"] == "moderate"

    def test_low_bias_detection(self):
        """Top 20% producing ≤60% of results → 'low' bias."""
        # 10 equal queries each contributing 10 new = 100 total
        query_stats = [(f"q{i}", 10, 10) for i in range(10)]
        result = sample_quality_report(
            enumerated=100, total=500, query_stats=query_stats
        )
        assert result["estimated_bias"] == "low"

    def test_query_efficiency_key_present(self):
        """query_efficiency must be present in the return dict."""
        query_stats = [("a", 100, 80)]
        result = sample_quality_report(enumerated=80, total=100, query_stats=query_stats)
        assert "query_efficiency" in result

    def test_query_efficiency_value(self):
        """
        query_efficiency = sum(new_count) / sum(result_count) when result_count > 0.

        Here: 60 new from 100 results = 0.60 efficiency.
        """
        query_stats = [
            ("a", 60, 40),   # 60 results, 40 new
            ("b", 40, 20),   # 40 results, 20 new
        ]
        result = sample_quality_report(enumerated=60, total=200, query_stats=query_stats)
        assert result["query_efficiency"] == pytest.approx(0.60)

    def test_result_contains_required_keys(self):
        """Return dict must contain coverage_pct, estimated_bias, query_efficiency."""
        query_stats = [("x", 10, 8)]
        result = sample_quality_report(enumerated=8, total=50, query_stats=query_stats)
        assert {"coverage_pct", "estimated_bias", "query_efficiency"}.issubset(result.keys())

    def test_empty_query_stats(self):
        """Empty query_stats list must not raise; efficiency defaults to 0.0."""
        result = sample_quality_report(enumerated=0, total=100, query_stats=[])
        assert result["query_efficiency"] == 0.0
        assert result["coverage_pct"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TestFormatStatsSummary
# ---------------------------------------------------------------------------

class TestFormatStatsSummary:
    """Tests for format_stats_summary(bot_rate_result, quality_report)."""

    def _make_inputs(self, flagged=30, total_sub=1000, coverage=75.0, bias="low"):
        """Produce realistic inputs for the formatter."""
        scored = _make_scored(flagged, 100)
        bot_rate = estimate_bot_rate(scored, total_subscribers=total_sub)
        quality = {
            "coverage_pct": coverage,
            "estimated_bias": bias,
            "query_efficiency": 0.72,
        }
        return bot_rate, quality

    def test_returns_non_empty_string(self):
        bot_rate, quality = self._make_inputs()
        out = format_stats_summary(bot_rate, quality)
        assert isinstance(out, str)
        assert len(out) > 0

    def test_contains_percentage(self):
        """Output must include a percentage representation of the bot rate."""
        bot_rate, quality = self._make_inputs(flagged=30)
        out = format_stats_summary(bot_rate, quality)
        # Should mention "30%" or "30.0%" somewhere
        assert "%" in out

    def test_contains_ci_bounds(self):
        """Output must reference both CI bounds."""
        bot_rate, quality = self._make_inputs(flagged=30)
        out = format_stats_summary(bot_rate, quality)
        # CI values should appear as percentages
        assert "CI" in out or "confidence" in out.lower() or "interval" in out.lower()

    def test_contains_coverage(self):
        """Output must include coverage information."""
        bot_rate, quality = self._make_inputs(coverage=75.0)
        out = format_stats_summary(bot_rate, quality)
        assert "75" in out or "coverage" in out.lower()

    def test_contains_bias(self):
        """Output must reference bias level."""
        bot_rate, quality = self._make_inputs(bias="high")
        out = format_stats_summary(bot_rate, quality)
        assert "high" in out.lower()

    def test_multiline(self):
        """Output should contain at least 2 lines."""
        bot_rate, quality = self._make_inputs()
        out = format_stats_summary(bot_rate, quality)
        assert "\n" in out
