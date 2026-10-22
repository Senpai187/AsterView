"""Local reusable report sign-off identity; no patient information is stored."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

ROLES = {"resident": "Student / Resident / Graduate student",
         "faculty": "Faculty / Consultant"}


def profile_path():
    from PySide6.QtCore import QStandardPaths
    return Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.GenericDataLocation)) / "AsterView" / "clinician-profile.json"


def validate_profile(profile):
    if not isinstance(profile, dict) or set(profile) != {"role", "name"}:
        raise ValueError("Choose your report role and name.")
    role, name = profile["role"], profile["name"]
    if not isinstance(role, str) or role not in ROLES or not isinstance(name, str) or not name.strip() or len(name) > 160 or any(ord(c) < 32 for c in name):
        raise ValueError("Choose a role and enter a name of up to 160 characters.")
    return {"role": role, "name": name.strip()}


def load_profile(path=None):
    target = profile_path() if path is None else Path(path)
    try:
        if target.stat().st_size > 4096:
            return None
        return validate_profile(json.loads(target.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError, KeyError):
        return None


def save_profile(profile, path=None):
    profile = validate_profile(profile)
    target = profile_path() if path is None else Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".aster-profile-", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(profile, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return profile


def ensure_clinician_profile(parent=None, *, force=False, path=None):
    """Startup/settings hook. First use asks once; cancelling leaves it unset."""
    previous = load_profile(path)
    if previous and not force:
        return previous
    from PySide6.QtCore import Qt, QSize
    import qtawesome as qta
    from PySide6.QtWidgets import (QDialog, QVBoxLayout, QLabel, QHBoxLayout,
                                   QPushButton, QButtonGroup, QLineEdit, QDialogButtonBox)
    dialog = QDialog(parent)
    dialog.setWindowTitle("Your report profile")
    dialog.setMinimumWidth(380)
    layout = QVBoxLayout(dialog)
    title = QLabel("How should your reports be signed?")
    title.setWordWrap(True)
    layout.addWidget(title)
    description = QLabel("Your role and name stay on this computer and can be changed in Report settings.")
    description.setWordWrap(True)
    layout.addWidget(description)
    roles = QButtonGroup(dialog)
    row = QHBoxLayout()
    buttons = {}
    for key, text in ROLES.items():
        button = QPushButton(text)
        button.setIcon(qta.icon("fa6s.user-tie" if key == "faculty" else "fa6s.graduation-cap", color="#d8b78a"))
        button.setIconSize(QSize(28, 28))
        button.setCheckable(True)
        button.setMinimumHeight(64)
        roles.addButton(button)
        buttons[key] = button
        row.addWidget(button)
    layout.addLayout(row)
    buttons[previous["role"] if previous else "resident"].setChecked(True)
    name = QLineEdit(previous["name"] if previous else "")
    name.setMaxLength(160)
    name.setPlaceholderText("Your name")
    layout.addWidget(name)
    error = QLabel()
    error.setWordWrap(True)
    error.setTextFormat(Qt.TextFormat.PlainText)
    layout.addWidget(error)
    actions = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
    layout.addWidget(actions)
    result = []
    def accept():
        try:
            value = save_profile({"role": next(key for key, button in buttons.items() if button.isChecked()),
                                  "name": name.text()}, path)
            result.append(value)
            dialog.accept()
        except (ValueError, OSError) as exc:
            error.setText(str(exc))
    actions.accepted.connect(accept)
    actions.rejected.connect(dialog.reject)
    dialog.exec()
    return result[0] if result else previous
