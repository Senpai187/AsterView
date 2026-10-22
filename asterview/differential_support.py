"""Small referenced reading aid driven only by reviewer-confirmed features.

No image classification, probabilities, diagnoses, ranking or exclusion is
performed. A displayed reference is not copied into a report without an
explicit clinician action. The reference set is deliberately nonexhaustive.
"""
from __future__ import annotations

FEATURES = {
    "periapical": "Periapical relationship confirmed",
    "nonvital_confirmed": "Nonvitality confirmed by pulp vitality testing",
    "pericoronal": "Pericoronal relationship confirmed",
    "well_defined": "Well-defined margin confirmed",
    "multilocular": "Multilocular appearance confirmed",
    "expansion": "Expansion confirmed on serial sections",
    "root_resorption": "Root resorption confirmed",
    "maxillary": "Maxillary location confirmed",
    "ill_defined": "Ill-defined/destructive margin confirmed",
}

REFERENCES = {
    "periapical_histology": {
        "title": "Rosenberg et al., J Endod 2010 · CBCT versus histopathology",
        "url": "https://pubmed.ncbi.nlm.nih.gov/20171356/",
        "consideration": "For a reviewer-confirmed periapical lesion, an inflammatory periapical process may be considered in the clinical context. This tool cannot distinguish a radicular cyst from a periapical granuloma.",
        "limit": "Original 45-patient comparative study; image appearance alone did not reliably separate cysts and granulomas. Correlate with pulp vitality testing and clinical findings.",
    },
    "maxillary_overlap": {
        "title": "Meng et al., DMFR 2019 · maxillary CT/CBCT lesion series",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6747442/",
        "consideration": "In an appropriate reviewer-confirmed maxillary cystic/pericoronal lesion, dentigerous cyst, odontogenic keratocyst and ameloblastoma are overlapping imaging considerations; no ranking is supplied.",
        "limit": "Original retrospective maxillary series. These considerations are not exhaustive and must not be generalized to another site or treated as histologic diagnoses.",
    },
    "odontogenic_overlap": {
        "title": "Kitisubkanchana et al., Oral Radiol 2021 · OKC and ameloblastoma radiographs",
        "url": "https://pubmed.ncbi.nlm.nih.gov/32030659/",
        "consideration": "Ameloblastoma and odontogenic keratocyst can overlap in radiographic appearance. Consider the full location, tooth relationship, internal structure and effects on adjacent anatomy before entering a differential.",
        "limit": "Original radiographic series of 100 OKCs and 101 ameloblastomas; not a validated CBCT classifier, disease probability or exhaustive differential.",
    },
}


def validate_features(features):
    if not isinstance(features, (list, tuple)) or len(features) > len(FEATURES) or any(not isinstance(item, str) or item not in FEATURES for item in features) or len(set(features)) != len(features):
        raise ValueError("Morphology must contain distinct supported reviewer-confirmed features.")
    return list(features)


def suggestions(features):
    confirmed = set(validate_features(features))
    if not confirmed or "ill_defined" in confirmed:
        return []
    keys = []
    if "periapical" in confirmed:
        keys.append("periapical_histology")
    if "maxillary" in confirmed and confirmed & {"pericoronal", "multilocular", "expansion"}:
        keys.append("maxillary_overlap")
    if confirmed & {"multilocular", "root_resorption"}:
        keys.append("odontogenic_overlap")
    return [{"id": key, **REFERENCES[key]} for key in keys]


def validate_support(ids):
    if not isinstance(ids, (list, tuple)) or len(ids) > len(REFERENCES) or any(not isinstance(key, str) or key not in REFERENCES for key in ids) or len(set(ids)) != len(ids):
        raise ValueError("Differential support must use distinct known reference identifiers.")
    return list(ids)
