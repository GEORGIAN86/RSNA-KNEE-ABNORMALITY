"""Canonical labels and MRI slot definitions shared by every pipeline stage."""

LABELS = (
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
)

SLOTS = (
    ("Sagittal", True),
    ("Sagittal", False),
    ("Coronal", True),
    ("Coronal", False),
    ("Axial", True),
    ("Axial", False),
)

CACHE_SCHEMA_VERSION = 1

__all__ = ["CACHE_SCHEMA_VERSION", "LABELS", "SLOTS"]
