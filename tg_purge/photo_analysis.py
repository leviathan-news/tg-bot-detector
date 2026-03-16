"""
Photo quality extraction from Telegram stripped thumbnails.

Telegram embeds a tiny JPEG thumbnail (~100-200 bytes) directly in the
UserProfilePhoto.stripped_thumb field. This module decodes that thumbnail
and computes quality metrics without any extra API calls.

The metrics map directly to the photo_quality features expected by
features.py:extract_features():
  - photo_file_size: stripped thumb byte count (proxy for complexity)
  - photo_edge_std: edge density standard deviation (texture detection)
  - photo_lum_variance: luminance variance (flat = synthetic/generated)
  - photo_sat_mean: mean saturation (oversaturated = common in bot avatars)
"""

import math
from collections import Counter
from typing import Dict, Optional


# Telegram stripped thumbnail JPEG reconstruction headers.
# The stripped_thumb field contains the JPEG body; these headers/footers
# complete it into a valid JPEG file for decoding.
_JPEG_HEADER = bytes([
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
    0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8,
])

_JPEG_FOOTER = bytes([0xFF, 0xD9])


def _decode_stripped_thumb(stripped_bytes):
    """Reconstruct and decode a Telegram stripped thumbnail into a PIL Image.

    Returns PIL Image or None if decoding fails or Pillow is not installed.
    """
    try:
        from PIL import Image
        import io
    except ImportError:
        return None

    if not stripped_bytes or len(stripped_bytes) < 3:
        return None

    # Reconstruct the JPEG: header + stripped_bytes[3:] + footer.
    # The first 3 bytes of stripped_bytes overlap with the header.
    jpeg_data = _JPEG_HEADER + stripped_bytes[3:] + _JPEG_FOOTER

    try:
        return Image.open(io.BytesIO(jpeg_data))
    except Exception:
        return None


def extract_photo_quality(user) -> Optional[Dict[str, float]]:
    """Extract photo quality metrics from a User's stripped thumbnail.

    No API calls — uses the stripped_thumb bytes embedded in
    user.photo (UserProfilePhoto). Returns None if the user has no photo
    or no stripped thumbnail, or if Pillow/numpy are unavailable.

    Returns dict matching features.py's photo_quality parameter:
        {photo_file_size, photo_edge_std, photo_lum_variance, photo_sat_mean}
    """
    photo = getattr(user, "photo", None)
    if photo is None:
        return None

    stripped = getattr(photo, "stripped_thumb", None)
    if not stripped:
        return None

    # Decode the thumbnail.
    img = _decode_stripped_thumb(stripped)
    if img is None:
        return None

    try:
        import numpy as np
    except ImportError:
        return None

    arr = np.array(img)
    result = {
        "photo_file_size": float(len(stripped)),
    }

    if len(arr.shape) == 3 and arr.shape[2] >= 3:
        # Luminance from RGB.
        gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        result["photo_lum_variance"] = float(np.var(gray))

        # Edge density (Sobel-like gradient).
        if gray.shape[0] > 2 and gray.shape[1] > 2:
            gx = np.diff(gray, axis=1)
            gy = np.diff(gray, axis=0)
            min_h = min(gx.shape[0], gy.shape[0])
            min_w = min(gx.shape[1], gy.shape[1])
            grad_mag = np.sqrt(gx[:min_h, :min_w] ** 2 + gy[:min_h, :min_w] ** 2)
            result["photo_edge_std"] = float(np.std(grad_mag))
        else:
            result["photo_edge_std"] = 0.0

        # Saturation (HSV-based).
        r = arr[:, :, 0] / 255.0
        g = arr[:, :, 1] / 255.0
        b = arr[:, :, 2] / 255.0
        cmax = np.maximum(np.maximum(r, g), b)
        cmin = np.minimum(np.minimum(r, g), b)
        delta = cmax - cmin
        sat = np.where(cmax > 0, delta / cmax, 0)
        result["photo_sat_mean"] = float(np.mean(sat))
    else:
        # Grayscale fallback.
        gray = arr if len(arr.shape) == 2 else arr[:, :, 0]
        result["photo_lum_variance"] = float(np.var(gray))
        result["photo_edge_std"] = 0.0
        result["photo_sat_mean"] = 0.0

    return result
