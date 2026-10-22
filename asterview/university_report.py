"""Editable, source-bound university report and lossless template-slot export.

The bundled blank form contains no case material. No image interpretation is
performed here: only explicitly accepted findings and entered text are used.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
from xml.dom import minidom
from zipfile import ZipFile

from .presentation import patient_display_name


REPORT_FORMAT = "asterview.university-report.v2"
LEGACY_REPORT_FORMAT = "asterview.university-report.v1"
SECTIONS = {
    "dentoalveolar": "Dentoalveolar Findings",
    "jaws": "Maxilla & Mandible",
    "sinuses": "Maxillary Sinuses & Nasal Cavity",
    "tmj": "Temporomandibular Joints",
    "cervical": "Cervical Spine",
    "airway": "Airway",
    "soft_tissues": "Soft Tissues",
    "other": "Other Findings",
}
FIELDS = {
    "patient_name": "Patient Name",
    "patient_id": "File / MRN",
    "age": "Age",
    "referrer": "Referring Clinician / Department",
    "examination_date": "Examination Date",
    "region": "Region / Field of View",
    "voxel_size": "Slice thickness",
    "comparison": "Comparison / Previous Imaging",
    "image_quality": "Image Quality",
    "impression": "Impression",
    "recommendations": "Recommendations",
    "resident": "Resident / Graduate Student",
    "faculty": "Faculty / Consultant",
    "report_date": "Report date",
}
# Stable content-control IDs in the supplied blank form, not positional indexes.
SLOTS = {
    "patient_name": "510032939", "patient_id": "-2047828384",
    "referrer": "-1686500917", "examination_date": "831411825",
    "clinical_indication": "-1971888567", "region": "336191664",
    "voxel_size": "-2089065020", "comparison": "1128657005",
    "image_quality": "1658187462", "dentoalveolar": "-157239261",
    "jaws": "-1434283900", "sinuses": "-1192839009", "tmj": "1002397894",
    "cervical": "-1576115108", "airway": "-1874525722",
    "soft_tissues": "502398755", "other": "1159424785",
    "impression": "-1613898592", "recommendations": "-1737545713",
    "resident": "1041087628", "faculty": "-861285801", "report_date": "-1899738361",
}
TEMPLATE_PATH = Path(__file__).with_name("assets") / "university-report-template.docx"
WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def validate_report(report, source=None):
    from .reporting import _text
    from .session import _keys
    _keys(report, {"format", "fields", "sections", "basis_token", "finding_ids"},
          "University report", {"profile", "concise_findings", "attachments"})
    if report["format"] not in {REPORT_FORMAT, LEGACY_REPORT_FORMAT}:
        raise ValueError("Unsupported university report format.")
    _keys(report["fields"], set(FIELDS), "University report fields")
    _keys(report["sections"], set(SECTIONS), "University report sections")
    fields = {key: _text(report["fields"][key], label, 16000)
              for key, label in FIELDS.items()}
    sections = {key: _text(report["sections"][key], label, 32000)
                for key, label in SECTIONS.items()}
    token = report["basis_token"]
    if not isinstance(token, str) or len(token) != 64 or any(c not in "0123456789abcdef" for c in token):
        raise ValueError("The report needs a valid accepted-findings basis.")
    ids = report["finding_ids"]
    if not isinstance(ids, list) or len(ids) > 128:
        raise ValueError("Report finding links must be a unique bounded list.")
    ids = [_text(value, "Report finding ID", 128, required=True) for value in ids]
    if len(set(ids)) != len(ids):
        raise ValueError("Report finding links must be unique.")
    profile = report.get("profile", "full")
    if not isinstance(profile, str) or profile not in {"full", "concise"}:
        raise ValueError("Choose the full university or concise report profile.")
    from .report_images import validate_attachments
    return dict(format=REPORT_FORMAT, fields=fields, sections=sections,
                basis_token=token, finding_ids=ids, profile=profile,
                concise_findings=_text(report.get("concise_findings", "\n".join(value for value in sections.values() if value.strip())),
                                       "Concise findings", 128000),
                attachments=validate_attachments(report.get("attachments", []), source))


def today_in_riyadh(now=None):
    """Saudi report date, independent of the computer's configured time zone."""
    try:
        from zoneinfo import ZoneInfo
        zone = ZoneInfo("Asia/Riyadh")
    except (ImportError, KeyError):
        zone = timezone(timedelta(hours=3), "Asia/Riyadh")
    moment = datetime.now(zone) if now is None else now.astimezone(zone)
    return moment.strftime("%d-%b-%Y")


def dicom_age(metadata):
    """Use recorded age, or completed years at examination from valid dates."""
    raw = str(metadata.get("patient_age", "")).strip()
    if len(raw) == 4 and raw[:3].isdigit() and raw[-1:] in "YMWD":
        number = int(raw[:3])
        return str(number) if raw[-1] == "Y" else f"{number} " + {"M": "months", "W": "weeks", "D": "days"}[raw[-1]]
    try:
        birth = datetime.strptime(str(metadata.get("patient_birth_date", "")), "%Y%m%d").date()
        exam = datetime.strptime(str(metadata.get("study_date", "")), "%Y%m%d").date()
        if exam >= birth:
            return str(exam.year - birth.year - ((exam.month, exam.day) < (birth.month, birth.day)))
    except ValueError:
        pass
    return ""


def source_slice_thickness(volume):
    """Recorded source slice thickness, or calibrated source-axis spacing."""
    raw = volume.metadata.get("source_slice_thickness_mm")
    try:
        value = float(raw) if not isinstance(raw, bool) else float("nan")
    except (ValueError, TypeError, OverflowError):
        value = float("nan")
    if not math.isfinite(value) or value <= 0:
        if not volume.metadata.get("geometry_calibrated", True):
            return ""
        value = float(volume.spacing[2])
    return f"{value:g} mm" if math.isfinite(value) and value > 0 else ""


def report_basis(document):
    """Changes to reviewed evidence or indication invalidate an exported draft."""
    payload = {"source": document["source"], "clinical_indication": document["clinical_indication"],
               "findings": [item for item in document["findings"] if item["review"] == "accepted"]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()


def create_report(document, volume, fingerprint, *, previous=None, clinician=None):
    from .reporting import validate_findings
    document = validate_findings(document, volume, fingerprint)
    metadata = volume.metadata
    if previous is not None:
        previous = validate_report(previous, document["source"])
        fields = previous["fields"]
    else:
        fields = dict.fromkeys(FIELDS, "")
        fields["patient_name"] = patient_display_name(str(metadata.get("patient_name", "")))
        fields["patient_id"] = str(metadata.get("patient_id", ""))
        fields["age"] = dicom_age(metadata)
        raw_date = str(metadata.get("study_date", ""))
        try:
            fields["examination_date"] = datetime.strptime(raw_date, "%Y%m%d").strftime("%d-%b-%Y")
        except ValueError:
            fields["examination_date"] = raw_date
        fields["voxel_size"] = source_slice_thickness(volume)
    fields["report_date"] = today_in_riyadh()
    if clinician is not None:
        from .clinician_profile import validate_profile
        clinician = validate_profile(clinician)
        fields["resident"] = clinician["name"] if clinician["role"] == "resident" else ""
        fields["faculty"] = clinician["name"] if clinician["role"] == "faculty" else ""
    accepted = [item for item in document["findings"] if item["review"] == "accepted"]
    grouped = {key: [] for key in SECTIONS}
    for item in accepted:
        grouped[item.get("report_section", "other")].append(item["observation"])
    fields["impression"] = "\n".join(item["differential"] for item in accepted if item["differential"].strip())
    return validate_report(dict(format=REPORT_FORMAT, fields=fields,
                                sections={key: "\n".join(lines) for key, lines in grouped.items()},
                                basis_token=report_basis(document),
                                finding_ids=[item["id"] for item in accepted],
                                profile=previous["profile"] if previous else ("concise" if clinician and clinician["role"] == "faculty" else "full"),
                                concise_findings="\n".join(item["observation"] for item in accepted),
                                attachments=previous["attachments"] if previous else []), document["source"])


def check_current_report(document, volume, fingerprint):
    from .reporting import validate_findings
    document = validate_findings(document, volume, fingerprint)
    report = document.get("university_report")
    if report is None or report["basis_token"] != report_basis(document):
        raise ValueError("Reviewed findings or clinical indication changed. Refresh the report from accepted findings before export; your edited draft remains saved.")
    accepted_ids = {item["id"] for item in document["findings"] if item["review"] == "accepted"}
    if set(report["finding_ids"]) != accepted_ids:
        raise ValueError("Report links do not match the current accepted findings.")
    return document, report


def _replace_control_text(dom, control, value):
    """Patch the existing text slot; retain cell, paragraph, run and SDT markup."""
    content = control.getElementsByTagName("w:sdtContent")[0]
    texts = content.getElementsByTagName("w:t")
    if len(texts) != 1:
        raise ValueError("The university template text slot has changed.")
    text = texts[0]
    text.setAttribute("xml:space", "preserve")
    while text.firstChild:
        text.removeChild(text.firstChild)
    # Word stores line breaks as elements. Input text and whitespace are retained.
    parts = value.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    text.appendChild(dom.createTextNode(parts[0]))
    cursor = text
    for line in parts[1:]:
        br = dom.createElementNS(WORD_NS, "w:br")
        cursor.parentNode.insertBefore(br, cursor.nextSibling)
        node = dom.createElementNS(WORD_NS, "w:t")
        node.setAttribute("xml:space", "preserve")
        node.appendChild(dom.createTextNode(line))
        br.parentNode.insertBefore(node, br.nextSibling)
        cursor = node
    for placeholder in list(control.getElementsByTagName("w:showingPlcHdr")):
        placeholder.parentNode.removeChild(placeholder)
    for style in list(content.getElementsByTagName("w:rStyle")):
        if style.getAttribute("w:val") == "PlaceholderText":
            style.parentNode.removeChild(style)


def render_docx_bytes(document, volume, fingerprint):
    """Fill a template copy; unmodified package entries retain identical bytes."""
    document, report = check_current_report(document, volume, fingerprint)
    slots = dict(report["fields"], **report["sections"],
                 clinical_indication=document["clinical_indication"])
    slots["report_date"] = today_in_riyadh()
    with ZipFile(TEMPLATE_PATH) as template:
        dom = minidom.parseString(template.read("word/document.xml"))
        controls = {}
        for control in dom.getElementsByTagName("w:sdt"):
            ids = control.getElementsByTagName("w:sdtPr")[0].getElementsByTagName("w:id")
            if ids:
                controls[ids[0].getAttribute("w:val")] = control
        if not set(SLOTS.values()).issubset(controls):
            raise ValueError("The university template is missing expected editable fields.")
        for field, identifier in SLOTS.items():
            _replace_control_text(dom, controls[identifier], slots[field])
        # Keep a subsection label with the narrative that follows it. The blank
        # source form did not need this, but filled tables may span pages.
        labels = {label + ":" for label in SECTIONS.values()}
        for paragraph in dom.getElementsByTagName("w:p"):
            visible = "".join(node.firstChild.data if node.firstChild else ""
                              for node in paragraph.getElementsByTagName("w:t"))
            if visible.strip() not in labels:
                continue
            properties = next((node for node in paragraph.childNodes if node.nodeName == "w:pPr"), None)
            if properties is None:
                properties = dom.createElementNS(WORD_NS, "w:pPr")
                paragraph.insertBefore(properties, paragraph.firstChild)
            if not properties.getElementsByTagName("w:keepNext"):
                properties.appendChild(dom.createElementNS(WORD_NS, "w:keepNext"))
        # The original Age field is a normal empty table cell, not a control.
        for text in dom.getElementsByTagName("w:t"):
            if text.firstChild and text.firstChild.data.strip() == "Age:":
                cell = text.parentNode.parentNode.parentNode
                siblings = [node for node in cell.parentNode.childNodes if node.nodeName == "w:tc"]
                index = siblings.index(cell)
                if index + 1 < len(siblings):
                    paragraph = siblings[index + 1].getElementsByTagName("w:p")[0]
                    run = dom.createElementNS(WORD_NS, "w:r")
                    node = dom.createElementNS(WORD_NS, "w:t")
                    node.appendChild(dom.createTextNode(slots["age"]))
                    run.appendChild(node)
                    paragraph.appendChild(run)
                break
        # Report remains a draft until the author signs it; no fabricated signer.
        for text in dom.getElementsByTagName("w:t"):
            if text.firstChild and text.firstChild.data == "CBCT INTERPRETATION REPORT":
                text.firstChild.data = "CBCT INTERPRETATION REPORT — DRAFT"
                break
        from .report_docx import compact_report, append_images
        by_key = {key: controls[identifier] for key, identifier in SLOTS.items()}
        end = compact_report(dom, by_key, slots, report, SECTIONS)
        media, relationships, content_types = append_images(
            dom, end, report["attachments"], template.read("word/_rels/document.xml.rels"),
            template.read("[Content_Types].xml"))
        settings = minidom.parseString(template.read("word/settings.xml"))
        updates = settings.getElementsByTagName("w:updateFields")
        update = updates[0] if updates else settings.createElementNS(WORD_NS, "w:updateFields")
        update.setAttribute("w:val", "true")
        if not updates:
            settings.documentElement.appendChild(update)
        patches = {"word/document.xml": dom.toxml(encoding="utf-8"),
                   "word/settings.xml": settings.toxml(encoding="utf-8")}
        if relationships is not None:
            patches["word/_rels/document.xml.rels"] = relationships
            patches["[Content_Types].xml"] = content_types
        stream = io.BytesIO()
        with ZipFile(stream, "w") as output:
            for entry in template.infolist():
                output.writestr(entry, patches.get(entry.filename, template.read(entry.filename)))
            for name, content in media.items():
                output.writestr(name, content)
        return stream.getvalue()


def export_docx(path, document, volume, fingerprint):
    """Atomic local export after validating exact source and current evidence."""
    content = render_docx_bytes(document, volume, fingerprint)
    destination = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=".aster-report-", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
