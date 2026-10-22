"""Template-derived report compaction and local inline image export."""
from __future__ import annotations

import base64
from xml.dom import minidom
from xml.sax.saxutils import escape

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
REL = "http://schemas.openxmlformats.org/package/2006/relationships"
IMAGE_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"


def _children(node, name):
    return [child for child in node.childNodes if child.nodeName == name]


def _text(node):
    return "".join(text.firstChild.data if text.firstChild else "" for text in node.getElementsByTagName("w:t"))


def _element(dom, name, **attributes):
    result = dom.createElementNS(W, "w:" + name)
    for key, value in attributes.items():
        result.setAttribute("w:" + key, str(value))
    return result


def _paragraph(dom, text, *, center=False, keep=False, page_break=False, size=20):
    p = _element(dom, "p")
    props = _element(dom, "pPr")
    props.appendChild(_element(dom, "spacing", after=120))
    if center:
        props.appendChild(_element(dom, "jc", val="center"))
    if keep:
        props.appendChild(_element(dom, "keepNext"))
    if page_break:
        props.appendChild(_element(dom, "pageBreakBefore"))
    p.appendChild(props)
    run = _element(dom, "r")
    rp = _element(dom, "rPr")
    rp.appendChild(_element(dom, "sz", val=size))
    run.appendChild(rp)
    for index, line in enumerate(text.replace("\r\n", "\n").replace("\r", "\n").split("\n")):
        if index:
            run.appendChild(_element(dom, "br"))
        node = _element(dom, "t")
        node.setAttribute("xml:space", "preserve")
        node.appendChild(dom.createTextNode(line))
        run.appendChild(node)
    p.appendChild(run)
    return p


def _replace_text(node, text):
    dom = node.ownerDocument
    paragraphs = node.getElementsByTagName("w:p")
    paragraph = paragraphs[0]
    old_runs = paragraph.getElementsByTagName("w:rPr")
    run_properties = old_runs[-1].cloneNode(True) if old_runs else None
    for child in list(paragraph.childNodes):
        if child.nodeName != "w:pPr":
            paragraph.removeChild(child)
    for old in list(paragraphs)[1:]:
        old.parentNode.removeChild(old)
    run = _element(dom, "r")
    if run_properties:
        run.appendChild(run_properties)
    for index, line in enumerate(text.replace("\r\n", "\n").replace("\r", "\n").split("\n")):
        if index:
            run.appendChild(_element(dom, "br"))
        value = _element(dom, "t")
        value.setAttribute("xml:space", "preserve")
        value.appendChild(dom.createTextNode(line))
        run.appendChild(value)
    paragraph.appendChild(run)


def _cell_width(cell, width, span=None):
    dom = cell.ownerDocument
    props = _children(cell, "w:tcPr")[0]
    for name in ("w:gridSpan", "w:tcW"):
        for old in _children(props, name):
            props.removeChild(old)
    props.insertBefore(_element(dom, "tcW", w=width, type="dxa"), props.firstChild)
    if span:
        props.appendChild(_element(dom, "gridSpan", val=span))


def _remove_block(node):
    following = node.nextSibling
    node.parentNode.removeChild(node)
    if following and following.nodeName == "w:p" and not _text(following).strip():
        following.parentNode.removeChild(following)


def _compact_metadata(table, fields, controls, slots):
    """Retain template header/cell styling; omit missing fields completely."""
    rows = _children(table, "w:tr")
    dom = table.ownerDocument
    header = rows[0].cloneNode(True)
    label_sample = _children(rows[1], "w:tc")[0]
    first_control = next((node for node in rows[1].childNodes if node.nodeName == "w:sdt"), None)
    value_sample = first_control.getElementsByTagName("w:tc")[0]
    entries = [(key, label) for key, label in fields if slots[key].strip()]
    if not entries:
        _remove_block(table)
        return
    for row in rows:
        table.removeChild(row)
    for grid in _children(table, "w:tblGrid"):
        table.removeChild(grid)
    grid = _element(dom, "tblGrid")
    grid.appendChild(_element(dom, "gridCol", w=2600))
    grid.appendChild(_element(dom, "gridCol", w=7525))
    table.appendChild(grid)
    _cell_width(header.getElementsByTagName("w:tc")[0], 10125, 2)
    table.appendChild(header)
    for key, label in entries:
        row = _element(dom, "tr")
        label_cell = label_sample.cloneNode(True)
        _cell_width(label_cell, 2600)
        _replace_text(label_cell, label + ":")
        row.appendChild(label_cell)
        if key in controls:
            value = controls[key].cloneNode(True)
            _cell_width(value.getElementsByTagName("w:tc")[0], 7525)
        else:
            value = value_sample.cloneNode(True)
            _cell_width(value, 7525)
            _replace_text(value, slots[key])
        row.appendChild(value)
        table.appendChild(row)


def compact_report(dom, controls_by_key, slots, report, sections):
    body = dom.getElementsByTagName("w:body")[0]
    tables = _children(body, "w:tbl")
    if len(tables) != 6:
        raise ValueError("The university template section layout has changed.")
    end = next((node for node in _children(body, "w:p") if "end of report" in _text(node).lower()), None)
    if end is None:
        raise ValueError("The university template is missing its report ending.")
    if report["profile"] == "concise":
        _remove_block(tables[0])
        _remove_block(tables[1])
        # Keep one generic Findings cell; remove all anatomy subsection labels.
        rows = _children(tables[2], "w:tr")
        value_row = rows[2]
        _replace_text(value_row, report["concise_findings"])
        for row in rows[1:]:
            if row is not value_row:
                tables[2].removeChild(row)
        if not report["concise_findings"].strip():
            _remove_block(tables[2])
    else:
        _compact_metadata(tables[0], [
            ("patient_name", "Patient Name"), ("patient_id", "File / MRN"), ("age", "Age"),
            ("referrer", "Referring Clinician / Department"), ("examination_date", "Examination Date"),
            ("clinical_indication", "Clinical Indication")], controls_by_key, slots)
        _compact_metadata(tables[1], [
            ("region", "Region / Field of View"), ("voxel_size", "Slice thickness"),
            ("comparison", "Comparison / Previous Imaging"), ("image_quality", "Image Quality")],
            controls_by_key, slots)
        rows = _children(tables[2], "w:tr")
        for index, key in enumerate(sections):
            if not slots[key].strip():
                tables[2].removeChild(rows[1 + index * 2])
                tables[2].removeChild(rows[2 + index * 2])
        if not any(slots[key].strip() for key in sections):
            _remove_block(tables[2])
    for table, field in ((tables[3], "impression"), (tables[4], "recommendations")):
        if not slots[field].strip():
            _remove_block(table)
    signers = [("resident", "Resident / Graduate Student"), ("faculty", "Faculty / Consultant")]
    if report["profile"] == "full" and any(slots[key].strip() for key, _ in signers):
        _compact_metadata(tables[5], signers + [("report_date", "Report date")], controls_by_key, slots)
    else:
        _remove_block(tables[5])
        # Concise reports have only three narrative section headings. A short
        # attribution/date line follows them without patient demographics.
        attribution = [label + ": " + slots[key] for key, label in signers if slots[key].strip()]
        attribution.append("Report date: " + slots["report_date"])
        body.insertBefore(_paragraph(dom, " · ".join(attribution), size=16), end)
    return end


def append_images(dom, end, attachments, relationships_bytes, content_types_bytes):
    """Append clean pixels and editable Word captions after sign-off."""
    if not attachments:
        return {}, None, None
    relationships = minidom.parseString(relationships_bytes)
    types = minidom.parseString(content_types_bytes)
    if not any(node.getAttribute("Extension").lower() == "png" for node in types.getElementsByTagName("Default")):
        default = types.createElementNS("http://schemas.openxmlformats.org/package/2006/content-types", "Default")
        default.setAttribute("Extension", "png")
        default.setAttribute("ContentType", "image/png")
        types.documentElement.appendChild(default)
    existing = {node.getAttribute("Id") for node in relationships.getElementsByTagName("Relationship")}
    body = end.parentNode
    media = {}
    for index, item in enumerate(attachments, 1):
        relation_id = f"rIdAsterReportImage{index}"
        if relation_id in existing:
            raise ValueError("The report image relationship already exists.")
        name = f"aster-report-{index}-{item['sha256'][:16]}.png"
        media["word/media/" + name] = base64.b64decode(item["png_base64"], validate=True)
        relation = relationships.createElementNS(REL, "Relationship")
        for key, value in {"Id": relation_id, "Type": IMAGE_REL, "Target": "media/" + name}.items():
            relation.setAttribute(key, value)
        relationships.documentElement.appendChild(relation)
        scale = min(6.85 * 914400 / item["width"], 4.25 * 914400 / item["height"])
        cx, cy = round(item["width"] * scale), round(item["height"] * scale)
        picture = f'''<w:drawing xmlns:w="{W}" xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
        <wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/>
        <wp:docPr id="{10000 + index}" name="AsterView image {index}" descr="{escape(item['title'], {'"': '&quot;'})}"/>
        <wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>
        <a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic>
        <pic:nvPicPr><pic:cNvPr id="{10000 + index}" name="{name}"/><pic:cNvPicPr/></pic:nvPicPr>
        <pic:blipFill><a:blip r:embed="{relation_id}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>
        <pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>
        <a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic>
        </wp:inline></w:drawing>'''
        paragraph = _paragraph(dom, "", center=True, keep=True)
        paragraph.getElementsByTagName("w:r")[0].appendChild(dom.importNode(minidom.parseString(picture).documentElement, True))
        body.insertBefore(paragraph, end)
        body.insertBefore(_paragraph(dom, item["caption"] or item["title"], center=True), end)
    return media, relationships.toxml(encoding="utf-8"), types.toxml(encoding="utf-8")
