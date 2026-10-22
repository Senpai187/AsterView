"""Readable display text without changing source metadata."""
import random


def patient_display_name(value):
    """Present explicit DICOM PN components given-name first; retain plain text."""
    groups = str(value or "").split("=")
    text = next((group for group in groups if group.replace("^", "").strip()), "")
    if "^" not in text:
        return " ".join(text.split())
    parts = (text.split("^") + [""] * 5)[:5]
    family, given, middle, prefix, suffix = parts
    return " ".join(" ".join((prefix, given, middle, family, suffix)).split())


RADIOGRAPH_JOKES = (
    "Keep calm and collimate.",
    "My humor is radiolucent.",
    "Radiographs: inner beauty, revealed.",
    "A little contrast makes life interesting.",
    "Good views. No skeletons in the closet.",
    "Let's look at this from another angle.",
    "A picture is worth a thousand voxels.",
)


def startup_joke(previous=None):
    """Avoid repeating the immediately preceding launch's line."""
    choices = [line for line in RADIOGRAPH_JOKES if line != previous]
    return random.choice(choices)
