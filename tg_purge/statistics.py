"""
Statistical sampling framework for bot-rate estimation and sample quality reporting.

All functions are pure math — no external dependencies, stdlib `math` only.
This module is intentionally decoupled from Telethon so it can be tested and
imported without a live Telegram connection.

Design notes
------------
* Wilson score interval is preferred over the simpler normal approximation
  because it remains well-behaved when p_hat is near 0 or 1, and for small n.
* Bias estimation uses the Pareto heuristic: if the top 20% of queries produce
  more than 80% of unique results, the sample is skewed ("high" bias).
* All public functions return plain Python dicts or primitive types — no
  dataclasses — so callers can easily serialise the results to JSON or CSV.
"""

import math


# ---------------------------------------------------------------------------
# Wilson score confidence interval
# ---------------------------------------------------------------------------

def wilson_score_interval(successes: int, total: int, z: float = 1.96) -> tuple:
    """Compute the Wilson score confidence interval for a proportion.

    The Wilson interval is more accurate than the normal-approximation ("Wald")
    interval, especially when the true proportion is near 0 or 1 or when the
    sample size is small.

    Formula (one-sided form combined into two bounds):
        p_hat = successes / total
        centre = (p_hat + z²/2n) / (1 + z²/n)
        half   = z * sqrt(p_hat*(1-p_hat)/n + z²/(4n²)) / (1 + z²/n)
        lower  = centre - half,  upper = centre + half

    Args:
        successes: Number of "positive" outcomes (e.g. flagged users).
        total:     Total sample size.
        z:         Critical value for the desired confidence level.
                   Default 1.96 corresponds to 95% CI.

    Returns:
        (lower, upper) as floats clamped to [0.0, 1.0].
        Returns (0.0, 0.0) when total == 0.
    """
    # Edge case: no data at all
    if total == 0:
        return (0.0, 0.0)

    p_hat = successes / total
    z2 = z * z
    n = total

    # Centre of the Wilson interval, shifted toward 0.5 by the z²/2n correction.
    centre = (p_hat + z2 / (2 * n)) / (1 + z2 / n)

    # Half-width: the radicand combines variance of p_hat with a bias-correction term.
    radicand = p_hat * (1 - p_hat) / n + z2 / (4 * n * n)
    # radicand is always >= 0 by construction, but guard against floating-point noise.
    half = z * math.sqrt(max(radicand, 0.0)) / (1 + z2 / n)

    # Clamp to unit interval and apply the zero-successes / all-successes edge cases.
    lower = max(centre - half, 0.0)
    upper = min(centre + half, 1.0)

    # Explicit edge-case overrides so callers can rely on exact 0.0 / 1.0 values.
    if successes == 0:
        lower = 0.0
    if successes == total:
        upper = 1.0

    return (lower, upper)


# ---------------------------------------------------------------------------
# Bot rate estimation
# ---------------------------------------------------------------------------

def estimate_bot_rate(
    scored_users: list,
    total_subscribers: int,
    threshold: int = 3,
) -> dict:
    """Estimate the bot/suspicious-account rate from a scored sample.

    Args:
        scored_users:      List of (user, score, reasons) tuples as returned by
                           scoring.score_user(). The 'user' element is not
                           inspected — only the integer score is used.
        total_subscribers: True subscriber count for the channel (used for
                           context; does not affect CI calculation, which is
                           purely sample-based).
        threshold:         Minimum score to count as flagged. Default 3.

    Returns:
        Dict with keys:
            point_estimate  — flagged / sample_size (0.0 if sample empty)
            ci_lower        — Wilson lower bound
            ci_upper        — Wilson upper bound
            margin_of_error — half-width of the CI: (ci_upper - ci_lower) / 2
            sample_size     — len(scored_users)
            flagged_count   — number of users at or above threshold
            total_subscribers — passed-through from the argument
    """
    sample_size = len(scored_users)

    if sample_size == 0:
        # No data: return a fully-zero dict so callers don't need a special code path.
        return {
            "point_estimate": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "margin_of_error": 0.0,
            "sample_size": 0,
            "flagged_count": 0,
            "total_subscribers": total_subscribers,
        }

    # Count users whose score is at or above the threshold.
    # scored_users entries are (user, score, reasons) — index 1 is the score.
    flagged_count = sum(1 for _user, score, _reasons in scored_users if score >= threshold)

    # Wilson CI on the flagged proportion.
    ci_lower, ci_upper = wilson_score_interval(flagged_count, sample_size)

    point_estimate = flagged_count / sample_size
    margin_of_error = (ci_upper - ci_lower) / 2

    return {
        "point_estimate": point_estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "margin_of_error": margin_of_error,
        "sample_size": sample_size,
        "flagged_count": flagged_count,
        "total_subscribers": total_subscribers,
    }


# ---------------------------------------------------------------------------
# Sample quality report
# ---------------------------------------------------------------------------

def sample_quality_report(
    enumerated: int,
    total: int,
    query_stats: list,
) -> dict:
    """Assess the quality and representativeness of an enumerated sample.

    Coverage is the straightforward ratio of enumerated unique users to the
    declared total subscriber count.

    Bias estimation uses a Pareto-style heuristic:
        - Sort queries by descending new_count (unique contributions).
        - Take the top 20% of queries (ceil to at least 1).
        - If those top queries produced > 80% of all new results → "high" bias.
        - If > 60% → "moderate" bias.
        - Else → "low" bias.
    This detects cases where a small number of prefix queries dominate the
    enumeration, which suggests the remaining population is under-sampled.

    Query efficiency is the global ratio of unique results to total API hits,
    measuring how much duplicate work the enumeration performed.

    Args:
        enumerated:   Number of unique users collected by the enumeration.
        total:        True subscriber count for the channel.
        query_stats:  List of (query, result_count, new_count) tuples where:
                          result_count — raw API results returned by this query
                          new_count    — how many of those were not seen before

    Returns:
        Dict with keys:
            coverage_pct      — (enumerated / total) * 100, or 0.0 if total == 0
            estimated_bias    — "high" | "moderate" | "low" | "unknown"
            query_efficiency  — sum(new_count) / sum(result_count), or 0.0 if no queries
    """
    # Coverage
    if total == 0:
        coverage_pct = 0.0
        estimated_bias = "unknown"
        query_efficiency = 0.0
        return {
            "coverage_pct": coverage_pct,
            "estimated_bias": estimated_bias,
            "query_efficiency": query_efficiency,
        }

    coverage_pct = (enumerated / total) * 100.0

    # Query efficiency: ratio of useful API hits to total API hits.
    total_results = sum(result_count for _q, result_count, _new in query_stats)
    total_new = sum(new_count for _q, _result, new_count in query_stats)

    if total_results == 0:
        query_efficiency = 0.0
    else:
        query_efficiency = total_new / total_results

    # Bias estimation via top-20% Pareto check.
    if not query_stats or total_new == 0:
        estimated_bias = "low"
    else:
        # Sort queries by their unique contribution, descending.
        sorted_by_new = sorted(query_stats, key=lambda x: x[2], reverse=True)

        # How many queries comprise the top 20% (at least 1).
        n_top = max(1, math.ceil(len(sorted_by_new) * 0.20))
        top_new = sum(new_count for _q, _r, new_count in sorted_by_new[:n_top])

        top_fraction = top_new / total_new
        if top_fraction > 0.80:
            estimated_bias = "high"
        elif top_fraction > 0.60:
            estimated_bias = "moderate"
        else:
            estimated_bias = "low"

    return {
        "coverage_pct": coverage_pct,
        "estimated_bias": estimated_bias,
        "query_efficiency": query_efficiency,
    }


# ---------------------------------------------------------------------------
# Terminal formatter
# ---------------------------------------------------------------------------

def format_stats_summary(bot_rate_result: dict, quality_report: dict) -> str:
    """Format bot rate and sample quality results for terminal display.

    Produces a multi-line human-readable summary suitable for printing
    directly to stdout or inserting into CLI command output.

    Args:
        bot_rate_result: Dict as returned by estimate_bot_rate().
        quality_report:  Dict as returned by sample_quality_report().

    Returns:
        Multi-line string. Does NOT add a trailing newline.
    """
    pe = bot_rate_result["point_estimate"] * 100
    ci_lo = bot_rate_result["ci_lower"] * 100
    ci_hi = bot_rate_result["ci_upper"] * 100
    moe = bot_rate_result["margin_of_error"] * 100
    sample = bot_rate_result["sample_size"]
    flagged = bot_rate_result["flagged_count"]
    total_sub = bot_rate_result["total_subscribers"]

    coverage = quality_report["coverage_pct"]
    bias = quality_report["estimated_bias"]
    efficiency = quality_report.get("query_efficiency", 0.0) * 100

    lines = [
        "=== Statistical Bot Rate Estimate ===",
        f"  Flagged:      {flagged}/{sample} sampled  ({pe:.1f}%)",
        f"  95% CI:       [{ci_lo:.1f}%, {ci_hi:.1f}%]  (±{moe:.1f}%)",
        f"  Total channel: {total_sub:>9,} subscribers",
        "",
        "=== Sample Quality ===",
        f"  Coverage:     {coverage:.1f}% of channel enumerated",
        f"  Bias:         {bias}",
        f"  Efficiency:   {efficiency:.1f}% unique results per API query",
    ]

    return "\n".join(lines)
