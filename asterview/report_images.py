"""Bounded local PNG attachments tied to an exact source fingerprint."""
from __future__ import annotations

import base64
import hashlib
import json
import struct
import uuid
import zlib

MAX_IMAGES = 16
MAX_IMAGE_BYTES = 6 * 1024 * 1024
MAX_TOTAL_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 24_000_000


def source_digest(source):
    from .session import _fingerprint
    source = _fingerprint(source, "Report image source")
    return hashlib.sha256(json.dumps(source, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()


def inspect_png(data):
    """Validate PNG framing and bounded dimensions before export/thumbnail decode."""
    if not isinstance(data, bytes) or not 45 <= len(data) <= MAX_IMAGE_BYTES or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Report images must be PNG files of at most 6 MiB.")
    offset, width, height, saw_data, ended = 8, None, None, False, False
    while offset + 12 <= len(data):
        size = struct.unpack(">I", data[offset:offset + 4])[0]
        kind = data[offset + 4:offset + 8]
        end = offset + 12 + size
        if end > len(data):
            raise ValueError("The report PNG is incomplete.")
        payload = data[offset + 8:end - 4]
        checksum = struct.unpack(">I", data[end - 4:end])[0]
        if zlib.crc32(kind + payload) & 0xffffffff != checksum:
            raise ValueError("The report PNG checksum is invalid.")
        if offset == 8:
            if kind != b"IHDR" or size != 13:
                raise ValueError("The report PNG has no valid image header.")
            width, height = struct.unpack(">II", payload[:8])
            if not 1 <= width <= 8192 or not 1 <= height <= 8192 or width * height > MAX_PIXELS:
                raise ValueError("The report PNG dimensions exceed the supported limit.")
        elif kind == b"IHDR":
            raise ValueError("The report PNG has duplicate image headers.")
        if kind == b"IDAT":
            saw_data = True
        if kind == b"IEND":
            if size or end != len(data):
                raise ValueError("The report PNG has invalid trailing data.")
            ended = True
            break
        offset = end
    if not saw_data or not ended:
        raise ValueError("The report PNG is missing image data.")
    return width, height


def make_attachment(png, title, source, *, caption="", view_state=None):
    width, height = inspect_png(png)
    item = {"id": str(uuid.uuid4()), "title": title, "caption": caption,
            "source_digest": source_digest(source), "sha256": hashlib.sha256(png).hexdigest(),
            "width": width, "height": height, "png_base64": base64.b64encode(png).decode("ascii"),
            "view_state": view_state or {}}
    return validate_attachments([item], source)[0]


def validate_attachments(items, source=None):
    from .reporting import _text
    from .session import _keys
    if not isinstance(items, list) or len(items) > MAX_IMAGES:
        raise ValueError("A report supports at most 16 selected images.")
    seen, checked, total = set(), [], 0
    expected_source = source_digest(source) if source is not None else None
    for item in items:
        _keys(item, {"id", "title", "caption", "source_digest", "sha256", "width",
                     "height", "png_base64", "view_state"}, "Report image")
        identifier = _text(item["id"], "Report image ID", 128, required=True)
        if identifier in seen:
            raise ValueError("Report image IDs must be unique.")
        seen.add(identifier)
        for key in ("source_digest", "sha256"):
            value = item[key]
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("A report image requires valid source and content hashes.")
        if expected_source is not None and item["source_digest"] != expected_source:
            raise ValueError("A report image belongs to a different source volume.")
        encoded = item["png_base64"]
        if not isinstance(encoded, str) or len(encoded) > (MAX_IMAGE_BYTES * 4 // 3 + 4):
            raise ValueError("The report image exceeds the supported size.")
        try:
            png = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("The report image encoding is invalid.") from exc
        width, height = inspect_png(png)
        if (type(item["width"]) is not int or type(item["height"]) is not int
                or (item["width"], item["height"]) != (width, height)
                or hashlib.sha256(png).hexdigest() != item["sha256"]):
            raise ValueError("The report image dimensions or hash changed.")
        total += len(png)
        if total > MAX_TOTAL_BYTES:
            raise ValueError("Selected report images must total at most 20 MiB.")
        # Capture metadata is plain bounded JSON, never code or filesystem paths.
        state = item["view_state"]
        if not isinstance(state, dict):
            raise ValueError("Report image view state must be an object.")
        try:
            encoded_state = json.dumps(state, allow_nan=False)
        except (ValueError, TypeError) as exc:
            raise ValueError("Report image view state must be finite JSON.") from exc
        if len(encoded_state) > 12000:
            raise ValueError("Report image view state is too large.")
        checked.append(dict(item, id=identifier,
                            title=_text(item["title"], "Report image title", 256, required=True),
                            caption=_text(item["caption"], "Report image caption", 4000),
                            view_state=json.loads(encoded_state)))
    return checked
