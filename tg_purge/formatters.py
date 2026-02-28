"""
Terminal, CSV, and JSON output formatting for tg-purge.

All output helpers write to stdout using print(). CSV and JSON export
use stdlib csv and json modules — no external dependencies.
"""

import csv
import io
import json
from collections import Counter

from .scoring import format_name, status_label, score_user


def print_section(title, scored_list, max_display=None):
    """Print a scored user list with header.

    Args:
        title: Section header text.
        scored_list: List of (user, score, reasons) tuples.
        max_display: Max items to display (None = all).
    """
    if not scored_list:
        return
    print(f"\n{'─' * 80}")
    print(f"  {title} ({len(scored_list)} users)")
    print(f"{'─' * 80}")

    display = scored_list[:max_display] if max_display else scored_list
    for user, s, reasons in display:
        name = format_name(user)
        status = status_label(user)
        verdict = "BOT?" if s >= 2 else "OK"
        reason_str = ", ".join(reasons) if reasons else "clean"
        print(f"[{verdict:4}] Score {s:2d} | {name:40s} | {status:15s} | {reason_str}")

    remaining = len(scored_list) - len(display)
    if remaining > 0:
        print(f"  ... and {remaining} more")


def print_score_distribution(scored_list, title="SCORE DISTRIBUTION"):
    """Print a histogram of score values.

    Args:
        scored_list: List of (user, score, reasons) tuples.
        title: Section header.
    """
    total = len(scored_list)
    if total == 0:
        return

    dist = Counter(s for _, s, _ in scored_list)

    print(f"\n{'─' * 80}")
    print(title)
    print(f"{'─' * 80}")
    for score_val in sorted(dist.keys()):
        count = dist[score_val]
        bar = "\u2588" * min(count, 60)
        pct = count / total * 100
        print(f"  Score {score_val:2d}: {count:4d} ({pct:5.1f}%) {bar}")


def print_signal_frequency(scored_list, title="SIGNAL FREQUENCY", top_n=20):
    """Print frequency of individual scoring signals.

    Args:
        scored_list: List of (user, score, reasons) tuples.
        title: Section header.
        top_n: Number of top signals to display.
    """
    total = len(scored_list)
    if total == 0:
        return

    signal_counts = Counter()
    for _, _, reasons in scored_list:
        for r in reasons:
            signal = r.split("(")[0]
            signal_counts[signal] += 1

    print(f"\n{'─' * 80}")
    print(title)
    print(f"{'─' * 80}")
    for signal, count in signal_counts.most_common(top_n):
        bar = "\u2588" * min(count, 40)
        print(f"  {signal:25s}: {count:4d} ({count/total*100:5.1f}%) {bar}")


def print_threshold_analysis(scored_list, title="THRESHOLD ANALYSIS"):
    """Print how many users would be flagged at each threshold.

    Args:
        scored_list: List of (user, score, reasons) tuples.
        title: Section header.
    """
    total = len(scored_list)
    if total == 0:
        return

    print(f"\n{'─' * 80}")
    print(title)
    print(f"{'─' * 80}")
    for threshold in [1, 2, 3, 4, 5]:
        flagged = sum(1 for _, s, _ in scored_list if s >= threshold)
        pct = flagged / total * 100
        print(f"  Threshold \u2265{threshold}: {flagged:5d} flagged ({pct:5.1f}%)")


def print_comparison_table(groups, title="COMPARISON"):
    """Print a side-by-side comparison of scored groups.

    Args:
        groups: List of (name, scored_list) tuples where scored_list
                contains (user, score, reasons) tuples.
        title: Section header.
    """
    if not groups:
        return

    print(f"\n{'=' * 80}")
    print(title)
    print(f"{'=' * 80}")

    # Header row
    col_width = 12
    header = f"{'':30s}"
    for name, _ in groups:
        header += f" {name:>{col_width}s}"
    print(f"\n{header}")

    labels = [
        ("Total", lambda scored: len(scored)),
        ("Score \u22652 %", lambda scored: f"{sum(1 for _, s, _ in scored if s >= 2)/max(len(scored),1)*100:.1f}%"),
        ("Score \u22653 %", lambda scored: f"{sum(1 for _, s, _ in scored if s >= 3)/max(len(scored),1)*100:.1f}%"),
    ]

    for label, fn in labels:
        row = f"  {label:28s}"
        for _, scored in groups:
            val = fn(scored)
            if isinstance(val, (int, float)):
                val = str(val)
            row += f" {val:>{col_width}s}"
        print(row)


def export_csv(scored_list, output_path):
    """Export scored users to CSV.

    Args:
        scored_list: List of (user, score, reasons) tuples.
        output_path: File path for CSV output.
    """
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "name", "username", "score", "status", "signals"])
        for user, score, reasons in scored_list:
            writer.writerow([
                user.id,
                format_name(user),
                user.username or "",
                score,
                status_label(user),
                "; ".join(reasons),
            ])
    print(f"Exported {len(scored_list)} users to {output_path}")


def export_json(scored_list, output_path):
    """Export scored users to JSON.

    Args:
        scored_list: List of (user, score, reasons) tuples.
        output_path: File path for JSON output.
    """
    entries = []
    for user, score, reasons in scored_list:
        entries.append({
            "user_id": user.id,
            "name": format_name(user),
            "username": user.username or None,
            "score": score,
            "status": status_label(user),
            "signals": reasons,
        })

    with open(output_path, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"Exported {len(entries)} users to {output_path}")
