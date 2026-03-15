#!/usr/bin/env python3
"""
Analyze profile photo characteristics of departed bots vs current subscribers.

Compares photo metadata and stripped thumbnail quality metrics between:
  - Users who left during the bot exodus (likely bots)
  - Current subscribers (mixed, but skewed human)

Metrics extracted WITHOUT downloading full photos:
  1. stripped_thumb size (bytes) — proxy for image complexity
  2. stripped_thumb entropy — measures information density
  3. photo_id distribution — detect reused/stock photos
  4. has_video flag — animated profiles (strong human signal)
  5. dc_id distribution — bot farms may cluster on specific data centers

For stripped thumbnails, also decodes the tiny JPEG to extract:
  - Luminance variance — flat/simple images have low variance
  - Color channel statistics — generated avatars often have unusual distributions
  - Edge density — real photos have more edges than simple graphics

Usage:
    python scripts/analyze_profile_photos.py --channel @leviathan_news --sample-size 200
    python scripts/analyze_profile_photos.py --channel @leviathan_news --departed-ids output/admin-log-departures-20260312.jsonl
"""

import argparse
import asyncio
import io
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from telethon import TelegramClient
from telethon.tl.functions.channels import GetParticipantsRequest
from telethon.tl.types import (
    ChannelParticipantsRecent,
    ChannelParticipantsSearch,
    UserProfilePhoto,
)

from tg_purge.config import load_config
from tg_purge.scoring import score_user


def decode_stripped_thumb(stripped_bytes):
    """Decode Telegram's stripped thumbnail format into a PIL Image.

    Telegram stores stripped thumbnails as a compact JPEG with a fixed
    header/footer. The format prepends/appends standard JPEG markers to
    the stored bytes. See MTProto docs: type `StrippedThumb`.

    Returns PIL Image or None if decoding fails.
    """
    try:
        from PIL import Image
    except ImportError:
        print("ERROR: Pillow is required. Install with: pip install Pillow", file=sys.stderr)
        sys.exit(1)

    if not stripped_bytes or len(stripped_bytes) < 3:
        return None

    # Telegram stripped thumb JPEG reconstruction.
    # The first byte after the header marker is the JPEG width/height.
    # Standard JPEG header for stripped thumbs:
    JPEG_HEADER = bytes([
        0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00,
        0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB,
        0x00, 0x43, 0x00, 0x28, 0x1C, 0x1E, 0x23, 0x1E, 0x19, 0x28, 0x23,
        0x21, 0x23, 0x2D, 0x2B, 0x28, 0x30, 0x3C, 0x64, 0x41, 0x3C, 0x37,
        0x37, 0x3C, 0x7B, 0x58, 0x5D, 0x49, 0x64, 0x91, 0x80, 0x99, 0x96,
        0x8F, 0x80, 0x8C, 0x8A, 0xA0, 0xB4, 0xE6, 0xC3, 0xA0, 0xAA, 0xDA,
        0xAD, 0x8A, 0x8C, 0xC8, 0xFF, 0xCB, 0xDA, 0xEE, 0xF5, 0xFF, 0xFF,
        0xFF, 0x9B, 0xC1, 0xFF, 0xFF, 0xFF, 0xFA, 0xFF, 0xE6, 0xFD, 0xFF,
        0xF8, 0xFF, 0xDB, 0x00, 0x43, 0x01, 0x2B, 0x2D, 0x2D, 0x3C, 0x35,
        0x3C, 0x76, 0x41, 0x41, 0x76, 0xF8, 0xA5, 0x8C, 0xA5, 0xF8, 0xF8,
        0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8,
        0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8,
        0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8,
        0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8,
        0xF8, 0xF8, 0xFF, 0xC0, 0x00, 0x11, 0x08, 0x00, 0x00, 0x00, 0x00,
        0x03, 0x01, 0x22, 0x00, 0x02, 0x11, 0x01, 0x03, 0x11, 0x01, 0xFF,
        0xC4, 0x00, 0x1F, 0x00, 0x00, 0x01, 0x05, 0x01, 0x01, 0x01, 0x01,
        0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
        0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0xFF,
        0xC4, 0x00, 0xB5, 0x10, 0x00, 0x02, 0x01, 0x03, 0x03, 0x02, 0x04,
        0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00, 0x01, 0x7D, 0x01, 0x02,
        0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13,
        0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08,
        0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0, 0x24, 0x33, 0x62,
        0x72, 0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26,
        0x27, 0x28, 0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A,
        0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55,
        0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
        0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x83,
        0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8A, 0x92, 0x93, 0x94, 0x95,
        0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7,
        0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9,
        0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2,
        0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2, 0xE3,
        0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4,
        0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFF, 0xC4, 0x00, 0x1F, 0x01,
        0x00, 0x03, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05,
        0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0xFF, 0xC4, 0x00, 0xB5, 0x11,
        0x00, 0x02, 0x01, 0x02, 0x04, 0x04, 0x03, 0x04, 0x07, 0x05, 0x04,
        0x04, 0x00, 0x01, 0x02, 0x77, 0x00, 0x01, 0x02, 0x03, 0x11, 0x04,
        0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71, 0x13,
        0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xA1, 0xB1, 0xC1, 0x09,
        0x23, 0x33, 0x52, 0xF0, 0x15, 0x62, 0x72, 0xD1, 0x0A, 0x16, 0x24,
        0x34, 0xE1, 0x25, 0xF1, 0x17, 0x18, 0x19, 0x1A, 0x26, 0x27, 0x28,
        0x29, 0x2A, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45,
        0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
        0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73,
        0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x82, 0x83, 0x84, 0x85,
        0x86, 0x87, 0x88, 0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97,
        0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9,
        0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2,
        0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4,
        0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6,
        0xE7, 0xE8, 0xE9, 0xEA, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
        0xF9, 0xFA, 0xFF, 0xDA, 0x00, 0x0C, 0x03, 0x01, 0x00, 0x02, 0x11,
        0x03, 0x11, 0x00, 0x3F, 0x00,
    ])
    JPEG_FOOTER = bytes([0xFF, 0xD9])

    # Patch dimensions into the header (bytes at offset 164 and 166)
    header = bytearray(JPEG_HEADER)
    header[164] = stripped_bytes[1]  # height
    header[166] = stripped_bytes[2]  # width

    # Build complete JPEG: header + payload (skip first 3 bytes) + footer
    jpeg_data = bytes(header) + stripped_bytes[3:] + JPEG_FOOTER

    try:
        img = Image.open(io.BytesIO(jpeg_data))
        img.load()  # Force decode
        return img
    except Exception:
        return None


def compute_image_metrics(img):
    """Compute quality metrics from a decoded thumbnail image.

    Returns a dict of metrics that characterize image complexity and quality.
    Low-quality or generated images tend to have:
      - Low luminance variance (flat colors)
      - Low edge density (no texture/detail)
      - Few unique colors (simple palette)
      - High color uniformity (dominated by one hue)
    """
    import numpy as np

    arr = np.array(img)

    metrics = {
        "width": img.width,
        "height": img.height,
        "pixels": img.width * img.height,
    }

    if len(arr.shape) == 3 and arr.shape[2] >= 3:
        # Convert to grayscale for luminance analysis
        gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]

        # Luminance variance — higher = more complex image
        metrics["luminance_mean"] = float(np.mean(gray))
        metrics["luminance_std"] = float(np.std(gray))
        metrics["luminance_variance"] = float(np.var(gray))

        # Color channel statistics
        for i, ch_name in enumerate(["r", "g", "b"]):
            metrics[f"{ch_name}_mean"] = float(np.mean(arr[:, :, i]))
            metrics[f"{ch_name}_std"] = float(np.std(arr[:, :, i]))

        # Color saturation (HSV-based) — bots often use oversaturated or desaturated images
        r, g, b = arr[:, :, 0] / 255.0, arr[:, :, 1] / 255.0, arr[:, :, 2] / 255.0
        cmax = np.maximum(np.maximum(r, g), b)
        cmin = np.minimum(np.minimum(r, g), b)
        delta = cmax - cmin
        # Saturation = delta / cmax (where cmax > 0)
        sat = np.where(cmax > 0, delta / cmax, 0)
        metrics["saturation_mean"] = float(np.mean(sat))
        metrics["saturation_std"] = float(np.std(sat))

        # Edge density — Sobel-like gradient magnitude
        # Simple horizontal and vertical gradient via numpy diff
        if gray.shape[0] > 2 and gray.shape[1] > 2:
            gx = np.diff(gray, axis=1)  # horizontal edges
            gy = np.diff(gray, axis=0)  # vertical edges
            # Trim to same shape
            min_h = min(gx.shape[0], gy.shape[0])
            min_w = min(gx.shape[1], gy.shape[1])
            grad_mag = np.sqrt(gx[:min_h, :min_w] ** 2 + gy[:min_h, :min_w] ** 2)
            metrics["edge_density_mean"] = float(np.mean(grad_mag))
            metrics["edge_density_std"] = float(np.std(grad_mag))
        else:
            metrics["edge_density_mean"] = 0.0
            metrics["edge_density_std"] = 0.0

        # Unique color count (quantized to 16-level per channel to avoid noise)
        quantized = (arr[:, :, :3] // 16).reshape(-1, 3)
        unique_colors = len(set(map(tuple, quantized)))
        metrics["unique_colors_q16"] = unique_colors
        # Normalized by pixel count
        metrics["color_diversity"] = unique_colors / max(metrics["pixels"], 1)

    else:
        # Grayscale image fallback
        gray = arr if len(arr.shape) == 2 else arr[:, :, 0]
        metrics["luminance_mean"] = float(np.mean(gray))
        metrics["luminance_std"] = float(np.std(gray))
        metrics["luminance_variance"] = float(np.var(gray))
        metrics["edge_density_mean"] = 0.0
        metrics["edge_density_std"] = 0.0
        metrics["unique_colors_q16"] = 0
        metrics["color_diversity"] = 0.0

    return metrics


def extract_photo_metadata(user):
    """Extract photo metadata from a Telethon User object without downloading.

    Returns a dict of metadata fields available from UserProfilePhoto, or
    None if the user has no photo.
    """
    photo = getattr(user, "photo", None)
    if photo is None:
        return None

    meta = {
        "has_photo": True,
        "photo_id": getattr(photo, "photo_id", None),
        "dc_id": getattr(photo, "dc_id", None),
        "has_video": bool(getattr(photo, "has_video", False)),
    }

    # stripped_thumb is the inline thumbnail (tiny JPEG, ~100-200 bytes)
    stripped = getattr(photo, "stripped_thumb", None)
    if stripped:
        meta["stripped_thumb_size"] = len(stripped)
        # Compute byte-level entropy of the raw thumbnail data
        byte_counts = Counter(stripped)
        total = len(stripped)
        entropy = -sum(
            (c / total) * math.log2(c / total)
            for c in byte_counts.values()
            if c > 0
        )
        meta["stripped_thumb_entropy"] = round(entropy, 4)

        # Decode and analyze the thumbnail image
        img = decode_stripped_thumb(stripped)
        if img:
            meta["image_metrics"] = compute_image_metrics(img)
        else:
            meta["image_metrics"] = None
    else:
        meta["stripped_thumb_size"] = 0
        meta["stripped_thumb_entropy"] = 0.0
        meta["image_metrics"] = None

    return meta


async def fetch_sample_subscribers(client, channel, count=200):
    """Fetch a sample of current subscribers using recent + search queries.

    Returns list of User objects.
    """
    users = {}

    # Recent subscribers
    result = await client(GetParticipantsRequest(
        channel=channel,
        filter=ChannelParticipantsRecent(),
        offset=0, limit=200, hash=0,
    ))
    for u in result.users:
        users[u.id] = u

    # A few search queries to get diverse samples
    for query in ["a", "m", "s", "d", "\u0430", "\u043c", ""]:
        if len(users) >= count:
            break
        result = await client(GetParticipantsRequest(
            channel=channel,
            filter=ChannelParticipantsSearch(query),
            offset=0, limit=200, hash=0,
        ))
        for u in result.users:
            users[u.id] = u
        await asyncio.sleep(1.0)

    return list(users.values())[:count]


async def fetch_departed_users(client, departed_ids, batch_size=100):
    """Fetch full User objects for departed user IDs.

    Many departed users may have deleted accounts, so we fetch what we can
    via client.get_entity(). Returns dict of user_id -> User.
    """
    from telethon.errors import UserNotParticipantError
    from telethon.tl.functions.users import GetUsersRequest
    from telethon.tl.types import InputUser

    users = {}
    id_list = list(departed_ids)

    for i in range(0, len(id_list), batch_size):
        batch = id_list[i:i + batch_size]
        for uid in batch:
            try:
                user = await client.get_entity(uid)
                users[uid] = user
            except Exception:
                pass  # User deleted or inaccessible
            await asyncio.sleep(0.3)

        print(
            f"  Fetched {min(i + batch_size, len(id_list))}/{len(id_list)} "
            f"departed users ({len(users)} accessible)",
            file=sys.stderr,
        )

    return users


def print_comparison(bot_metrics, human_metrics, label_a="BOTS", label_b="HUMANS"):
    """Print side-by-side statistical comparison of photo metrics."""
    import numpy as np

    print(f"\n{'=' * 80}")
    print(f"PROFILE PHOTO ANALYSIS: {label_a} vs {label_b}")
    print(f"{'=' * 80}")
    print(f"  {label_a}: {len(bot_metrics)} users with photos analyzed")
    print(f"  {label_b}: {len(human_metrics)} users with photos analyzed")

    # Compare metadata-level stats
    def compare_field(field, bot_data, human_data, fmt=".2f"):
        bot_vals = [d[field] for d in bot_data if field in d and d[field] is not None]
        human_vals = [d[field] for d in human_data if field in d and d[field] is not None]
        if not bot_vals or not human_vals:
            return
        b_mean, b_std = np.mean(bot_vals), np.std(bot_vals)
        h_mean, h_std = np.mean(human_vals), np.std(human_vals)
        diff_pct = ((b_mean - h_mean) / h_mean * 100) if h_mean != 0 else 0
        print(f"  {field:30s}  {label_a}: {b_mean:{fmt}} (+/-{b_std:{fmt}})  "
              f"{label_b}: {h_mean:{fmt}} (+/-{h_std:{fmt}})  "
              f"diff: {diff_pct:+.1f}%")

    print(f"\n{'─' * 80}")
    print("PHOTO METADATA")
    print(f"{'─' * 80}")

    # has_video comparison
    bot_video = sum(1 for d in bot_metrics if d.get("has_video"))
    human_video = sum(1 for d in human_metrics if d.get("has_video"))
    bot_video_pct = bot_video / max(len(bot_metrics), 1) * 100
    human_video_pct = human_video / max(len(human_metrics), 1) * 100
    print(f"  {'has_video':30s}  {label_a}: {bot_video_pct:.1f}% ({bot_video})  "
          f"{label_b}: {human_video_pct:.1f}% ({human_video})")

    compare_field("stripped_thumb_size", bot_metrics, human_metrics, ".0f")
    compare_field("stripped_thumb_entropy", bot_metrics, human_metrics)

    # DC distribution
    print(f"\n{'─' * 80}")
    print("DATA CENTER DISTRIBUTION")
    print(f"{'─' * 80}")
    bot_dcs = Counter(d.get("dc_id") for d in bot_metrics if d.get("dc_id"))
    human_dcs = Counter(d.get("dc_id") for d in human_metrics if d.get("dc_id"))
    all_dcs = sorted(set(bot_dcs.keys()) | set(human_dcs.keys()))
    for dc in all_dcs:
        b_pct = bot_dcs.get(dc, 0) / max(len(bot_metrics), 1) * 100
        h_pct = human_dcs.get(dc, 0) / max(len(human_metrics), 1) * 100
        print(f"  DC {dc}:  {label_a}: {b_pct:.1f}% ({bot_dcs.get(dc, 0)})  "
              f"{label_b}: {h_pct:.1f}% ({human_dcs.get(dc, 0)})")

    # Image metrics comparison (only for users with decoded thumbnails)
    bot_img = [d["image_metrics"] for d in bot_metrics
               if d.get("image_metrics") is not None]
    human_img = [d["image_metrics"] for d in human_metrics
                 if d.get("image_metrics") is not None]

    if bot_img and human_img:
        print(f"\n{'─' * 80}")
        print(f"IMAGE QUALITY METRICS (from stripped thumbnails)")
        print(f"  {label_a}: {len(bot_img)} thumbnails decoded")
        print(f"  {label_b}: {len(human_img)} thumbnails decoded")
        print(f"{'─' * 80}")

        for field in ["luminance_std", "luminance_variance", "saturation_mean",
                       "saturation_std", "edge_density_mean", "edge_density_std",
                       "unique_colors_q16", "color_diversity"]:
            compare_field(field, bot_img, human_img)

    # Photo ID reuse detection
    print(f"\n{'─' * 80}")
    print("PHOTO ID REUSE (shared photos across accounts)")
    print(f"{'─' * 80}")
    bot_photo_ids = [d["photo_id"] for d in bot_metrics if d.get("photo_id")]
    human_photo_ids = [d["photo_id"] for d in human_metrics if d.get("photo_id")]
    bot_id_counts = Counter(bot_photo_ids)
    human_id_counts = Counter(human_photo_ids)
    bot_dupes = sum(1 for c in bot_id_counts.values() if c > 1)
    human_dupes = sum(1 for c in human_id_counts.values() if c > 1)
    print(f"  {label_a}: {bot_dupes} photo IDs shared by multiple accounts")
    print(f"  {label_b}: {human_dupes} photo IDs shared by multiple accounts")
    if bot_dupes > 0:
        top_dupes = bot_id_counts.most_common(5)
        for pid, cnt in top_dupes:
            if cnt > 1:
                print(f"    photo_id {pid}: used by {cnt} bot accounts")


async def run(args):
    """Main async entry point."""
    config = load_config(args.config)
    if args.channel:
        config.default_channel = args.channel
    if args.session_path:
        config.session_path = args.session_path
    config.validate_credentials()

    client = TelegramClient(
        config.session_path,
        int(config.api_id),
        config.api_hash,
    )
    await client.start()
    me = await client.get_me()
    print(f"Connected as: {me.first_name}", file=sys.stderr)

    channel_id = args.channel or config.default_channel
    entity = await client.get_entity(channel_id)
    print(f"Channel: {entity.title}", file=sys.stderr)

    # ── Collect departed user IDs ──
    departed_ids = set()
    if args.departed_ids:
        # Load from JSONL file (admin log dump or monitor output)
        with open(args.departed_ids) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                uid = record.get("user_id")
                if uid:
                    departed_ids.add(int(uid))
        print(f"Loaded {len(departed_ids)} departed user IDs from {args.departed_ids}",
              file=sys.stderr)
    else:
        print("No --departed-ids file provided. Will only analyze current subscribers.",
              file=sys.stderr)

    # ── Sample current subscribers ──
    print(f"\nFetching {args.sample_size} current subscribers...", file=sys.stderr)
    current_users = await fetch_sample_subscribers(client, entity, args.sample_size)
    print(f"Got {len(current_users)} current subscribers", file=sys.stderr)

    # ── Analyze current subscribers (our "human-leaning" sample) ──
    human_metrics = []
    no_photo_humans = 0
    for user in current_users:
        meta = extract_photo_metadata(user)
        if meta:
            human_metrics.append(meta)
        else:
            no_photo_humans += 1

    print(f"Current subscribers: {len(human_metrics)} with photos, "
          f"{no_photo_humans} without", file=sys.stderr)

    # ── Analyze departed users (our "bot" sample) ──
    bot_metrics = []
    no_photo_bots = 0

    if departed_ids:
        # Try to get user objects for departed users
        # First check if the JSONL has profile data embedded
        profiles_from_jsonl = {}
        if args.departed_ids:
            with open(args.departed_ids) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    uid = record.get("user_id")
                    profile = record.get("profile")
                    if uid and profile:
                        profiles_from_jsonl[int(uid)] = profile

        if profiles_from_jsonl:
            # JSONL profiles don't have stripped_thumb — need to fetch live
            print(f"\nJSONL has {len(profiles_from_jsonl)} profiles but no thumbnail data.",
                  file=sys.stderr)
            print(f"Need to fetch live user objects for photo analysis.", file=sys.stderr)

        # Sample departed users (don't fetch all if there are thousands)
        sample_departed = list(departed_ids)[:args.sample_size]
        print(f"\nFetching {len(sample_departed)} departed user profiles...", file=sys.stderr)

        for i, uid in enumerate(sample_departed):
            try:
                user = await client.get_entity(uid)
                meta = extract_photo_metadata(user)
                if meta:
                    # Also add the heuristic score for context
                    score, reasons = score_user(user)
                    meta["heuristic_score"] = score
                    meta["reasons"] = reasons
                    bot_metrics.append(meta)
                else:
                    no_photo_bots += 1
            except Exception:
                no_photo_bots += 1  # Deleted/inaccessible = no photo

            if (i + 1) % 50 == 0:
                print(f"  ...{i + 1}/{len(sample_departed)} fetched "
                      f"({len(bot_metrics)} with photos)", file=sys.stderr)
            await asyncio.sleep(0.3)

        print(f"Departed users: {len(bot_metrics)} with photos, "
              f"{no_photo_bots} without", file=sys.stderr)

    # ── No-photo rate comparison ──
    print(f"\n{'=' * 80}")
    print("NO-PHOTO RATE")
    print(f"{'=' * 80}")
    total_bots = len(bot_metrics) + no_photo_bots
    total_humans = len(human_metrics) + no_photo_humans
    if total_bots:
        print(f"  BOTS:   {no_photo_bots}/{total_bots} "
              f"({no_photo_bots / total_bots * 100:.1f}%) have no photo")
    if total_humans:
        print(f"  HUMANS: {no_photo_humans}/{total_humans} "
              f"({no_photo_humans / total_humans * 100:.1f}%) have no photo")

    # ── Full comparison ──
    if bot_metrics and human_metrics:
        print_comparison(bot_metrics, human_metrics)

        # Save raw metrics for further analysis
        output_path = args.output or "output/photo-analysis.json"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "channel": str(channel_id),
                "bot_sample_size": len(bot_metrics),
                "human_sample_size": len(human_metrics),
                "bot_metrics": bot_metrics,
                "human_metrics": human_metrics,
                "no_photo_bots": no_photo_bots,
                "no_photo_humans": no_photo_humans,
            }, f, indent=2, default=str)
        print(f"\nRaw metrics saved to: {output_path}", file=sys.stderr)
    elif not bot_metrics:
        print("\nNo bot photos to analyze. Provide --departed-ids with a JSONL file.",
              file=sys.stderr)

    await client.disconnect()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze profile photo quality: departed bots vs current subscribers."
    )
    parser.add_argument(
        "--channel", required=True,
        help="Target channel username or numeric ID.",
    )
    parser.add_argument(
        "--departed-ids", dest="departed_ids", default=None,
        help="JSONL file of departed users (from admin log dump or monitor).",
    )
    parser.add_argument(
        "--sample-size", dest="sample_size", type=int, default=200,
        help="Number of users to sample per group (default: 200).",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON file for raw metrics (default: output/photo-analysis.json).",
    )
    parser.add_argument("--config", default=None, help="TOML config file path.")
    parser.add_argument(
        "--session-path", dest="session_path", default=None,
        help="Override session file path.",
    )
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
