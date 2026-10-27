#!/usr/bin/env python3
"""Audit I-SPY1 original DCE phases without modifying source DICOM files.

The first post-contrast state is labeled 1 after a verified pre-contrast state;
its observed acquisition index can be larger when pre scans are repeated. The
UCSF SER early-post index can also be later and is not substituted for the true
first post. Every volume retains spatially ordered source instances and their
original patient geometry.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import csv
import json
import math
import os
from pathlib import Path
import re
import warnings

import numpy as np
import pydicom

warnings.filterwarnings("ignore", module="pydicom")


def js(value):
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip("\0 ")
    try:
        return [js(x) for x in value]
    except TypeError:
        return str(value)


def private(ds, suffix):
    elem = ds.get((0x0117, suffix))
    return js(elem.value) if elem is not None else None


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    tmp.replace(path)


def dcm_files(root, meta):
    directory = root / "images" / meta["PatientID"] / meta["StudyInstanceUID"] / meta["SeriesInstanceUID"]
    return sorted(str(p) for p in directory.glob("*.dcm"))


def read_header(path):
    ds = pydicom.dcmread(path, stop_before_pixels=True)
    keys = ["SOPInstanceUID", "SeriesNumber", "SeriesDescription", "ImageType", "Modality",
            "Manufacturer", "Rows", "Columns", "PixelSpacing", "ImageOrientationPatient",
            "ImagePositionPatient", "SpacingBetweenSlices", "SliceThickness",
            "TemporalPositionIdentifier", "NumberOfTemporalPositions", "AcquisitionNumber",
            "AcquisitionTime", "ContentTime", "InstanceNumber", "SequenceName", "ProtocolName",
            "ScanningSequence", "MRAcquisitionType", "RepetitionTime", "EchoTime", "FlipAngle",
            "ClinicalTrialTimePointID", "ClinicalTrialTimePointDescription", "FrameOfReferenceUID"]
    keys += ["ContrastBolusAgent", "ContrastBolusStartTime"]
    result = {k: js(ds.get(k)) for k in keys}
    result["file"] = path
    result["ucsf"] = {"total_phases": private(ds, 0x1030),
                      "phase_duration": private(ds, 0x1031),
                      "start_delays": private(ds, 0x1032),
                      "injection_time": private(ds, 0x1033),
                      "effective_delays": private(ds, 0x1034),
                      "ser_phase_indices": private(ds, 0x1035),
                      "timing_method": private(ds, 0x103A),
                      "timing_comments": private(ds, 0x103B),
                      "protocol_compliant": private(ds, 0x10C5),
                      "protocol_comments": private(ds, 0x10C6)}
    return result


def representative(item):
    root, meta = item
    files = dcm_files(root, meta)
    try:
        header = read_header(files[0])
        return {"metadata": meta, "header": header, "nfiles": len(files)}
    except Exception as exc:
        return {"metadata": meta, "error": repr(exc), "nfiles": len(files)}


def is_original(s):
    h = s.get("header", {})
    typ = " ".join(h.get("ImageType") or []).upper()
    return h.get("Modality") == "MR" and "ORIGINAL" in typ and "DERIVED" not in typ


def excluded_reason(s):
    h = s.get("header", {})
    text = " ".join(str(h.get(x) or "") for x in ["SeriesDescription", "SequenceName", "ProtocolName"])
    if not is_original(s):
        return "not_original_MR"
    if re.search(r"PROJECTION|MIP|VELOCITY MAP", " ".join(h.get("ImageType") or []), re.I):
        return "projection_or_velocity_image_type"
    if re.search(r"T2|TSET2|TSE2|FSE|STIR|BOLD|DWI|DTI|DIFF|ADC|SCOUT|LOCAL|SURVEY|B0MAP|MIP|PJN|SUBTRA|(?:_|[ /-])SUB(?:$|[_ /])|PROJECTION|NEEDLE", text, re.I):
        return "non_DCE_or_projection_description"
    return None


def number(value, default=None):
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def clock_seconds(value):
    if not value:
        return None
    try:
        text = str(value).replace(":", "")
        # DICOM TM legally omits trailing components: HH, HHMM, HHMMSS.FF.
        whole, dot, fraction = text.partition(".")
        if len(whole) not in (2, 4, 6):
            return None
        hours = int(whole[:2])
        minutes = int(whole[2:4]) if len(whole) >= 4 else 0
        seconds = int(whole[4:6]) if len(whole) >= 6 else 0
        return hours*3600 + minutes*60 + seconds + (float("0."+fraction) if dot else 0)
    except (ValueError, TypeError):
        return None


def spatial_key(h):
    return tuple(round(float(x), 3) for x in (h.get("ImagePositionPatient") or []))


def volume_geometry(headers):
    reasons = []
    h0 = headers[0]
    try:
        orientation = np.asarray(h0["ImageOrientationPatient"], dtype=float)
        normal = np.cross(orientation[:3], orientation[3:])
        if abs(np.linalg.norm(normal) - 1) > 1e-3:
            reasons.append("non_orthonormal_orientation")
        normal /= np.linalg.norm(normal)
        ordered = sorted(headers, key=lambda h: float(np.dot(np.asarray(h["ImagePositionPatient"], float), normal)))
        positions = np.asarray([h["ImagePositionPatient"] for h in ordered], float)
        projected = positions @ normal
        gaps = np.diff(projected)
        spacing = float(np.median(gaps)) if len(gaps) else None
        if len(ordered) < 8:
            reasons.append("fewer_than_8_slices")
        if len(gaps) and (np.min(gaps) <= .01 or np.max(np.abs(gaps-spacing)) > .05):
            reasons.append("duplicate_or_nonuniform_slice_positions")
        if len(gaps) and np.max(np.linalg.norm(np.diff(positions, axis=0) - gaps[:, None]*normal, axis=1)) > .05:
            reasons.append("slice_positions_not_collinear")
        for h in ordered:
            if h.get("Rows") != h0.get("Rows") or h.get("Columns") != h0.get("Columns"):
                reasons.append("varying_matrix")
            if not np.allclose(h["ImageOrientationPatient"], orientation, atol=1e-5):
                reasons.append("varying_orientation")
            if not np.allclose(h["PixelSpacing"], h0["PixelSpacing"], atol=1e-4):
                reasons.append("varying_pixel_spacing")
        geo = {"rows": h0["Rows"], "columns": h0["Columns"],
               "pixel_spacing": h0["PixelSpacing"], "iop": orientation.tolist(),
               "positions": positions.tolist(), "slice_spacing": spacing,
               "origin_lps": positions[0].tolist(), "slice_direction": normal.tolist(),
               "frame_of_reference_uid": h0.get("FrameOfReferenceUID"),
               "coordinate_system": "DICOM_LPS", "array_axis_order": "slice,row,column"}
        return ordered, geo, sorted(set(reasons))
    except Exception as exc:
        return headers, {}, ["invalid_geometry:" + str(exc)]


def split_phases(headers):
    """Split temporal positions, or proven complete repeated spatial grids."""
    positions = Counter(spatial_key(h) for h in headers)
    if len(positions) == len(headers):
        return [headers], "single_volume"
    npositions = len(positions)
    repeats = set(positions.values())
    if len(repeats) != 1 or not npositions:
        raise ValueError("unbalanced_repeated_spatial_positions")
    tpi = defaultdict(list)
    for h in headers:
        tpi[number(h.get("TemporalPositionIdentifier"))].append(h)
    if None not in tpi and len(tpi) == next(iter(repeats)):
        groups = [tpi[k] for k in sorted(tpi)]
        method = "TemporalPositionIdentifier"
    else:
        if len({number(h.get("InstanceNumber")) for h in headers}) != len(headers):
            raise ValueError("instance_numbers_not_unique_for_temporal_split")
        ordered = sorted(headers, key=lambda h: number(h["InstanceNumber"]))
        groups = [ordered[i:i+npositions] for i in range(0, len(ordered), npositions)]
        method = "complete_spatial_grids_in_InstanceNumber_order"
    expected = set(positions)
    for group in groups:
        if len(group) != npositions or {spatial_key(h) for h in group} != expected:
            raise ValueError("temporal_group_is_not_one_complete_spatial_grid")
    medians = []
    for group in groups:
        times = [clock_seconds(h.get("AcquisitionTime") or h.get("ContentTime")) for h in group]
        times = [x for x in times if x is not None]
        medians.append(float(np.median(times)) if times else None)
    if all(x is not None for x in medians) and any(b < a for a, b in zip(medians, medians[1:])):
        raise ValueError("temporal_order_disagrees_with_acquisition_times")
    return groups, method


def description_family(text):
    text = str(text or "").lower()
    text = re.sub(r":.*$", "", text)
    text = re.sub(r"\+c\b|\(new\)|\buni\b|_pos\b|/immp\b|(?:\s|_)+[12]$", "", text)
    text = re.sub(r"\b(pre|post)\b|contrast|\bdel(?:ay(?:ed)?)?a?\b|phase\s*[=\-]?\s*\d+", "", text)
    return re.sub(r"[^a-z0-9]", "", text)


def related_series(anchor, candidates, allow_sequence_match=False):
    ah = anchor["header"]
    family = description_family(ah.get("SeriesDescription"))
    result = []
    for s in candidates:
        h = s["header"]
        same_family = description_family(h.get("SeriesDescription")) == family
        same_sequence = False
        if allow_sequence_match and h.get("SequenceName") and h.get("SequenceName") == ah.get("SequenceName"):
            try:
                same_sequence = all(abs(float(h[k])-float(ah[k])) <= max(.1, abs(float(ah[k]))*.01)
                                    for k in ["RepetitionTime", "EchoTime"])
            except (ValueError, TypeError, KeyError):
                pass
        if not same_family and not same_sequence:
            continue
        if h.get("Rows") != ah.get("Rows") or h.get("Columns") != ah.get("Columns"):
            continue
        try:
            if not allow_sequence_match and not np.allclose(h["PixelSpacing"], ah["PixelSpacing"], atol=.01):
                continue
            if not np.allclose(h["ImageOrientationPatient"], ah["ImageOrientationPatient"], atol=1e-3):
                continue
        except (ValueError, TypeError):
            continue
        result.append(s)
    return result


def explicit_prepost_anchor(candidates):
    """Recover only independently named pre and post volumes in one family.

    Repeated series all named POST are deliberately insufficient: several
    I-SPY1 noncompliant scans use that label for every stored acquisition.
    """
    valid = []
    for s in candidates:
        desc = str(s["header"].get("SeriesDescription") or "")
        if re.search(r"\bPOST\b", desc, re.I) and not re.search(r"\bPRE\b|NEEDLE", desc, re.I):
            family = related_series(s, candidates)
            pre = [a for a in family if not re.search(r"\bPOST\b", str(a["header"].get("SeriesDescription") or ""), re.I)]
            if len(pre) == 1:
                valid.append(pre[0])
    unique = {s["metadata"]["SeriesInstanceUID"]: s for s in valid}
    return next(iter(unique.values())) if len(unique) == 1 else None


def scan_study(task):
    root, study_key, series, cache_dir = task
    patient_id, study_uid = study_key
    case_id = patient_id + "_" + study_uid.rsplit(".", 1)[-1]
    record = {"case_id": case_id, "patient_id": patient_id, "study_uid": study_uid,
              "status": "review", "first_post": None, "qc": {"reasons": []}}
    known = [s["header"] for s in series if "header" in s]
    # TCIA SEG/SR use numeric IDs and MRI_n while original MR use Tn; use
    # original MR as the authoritative visit, keeping all header variants.
    all_visits = {(h.get("ClinicalTrialTimePointID"), h.get("ClinicalTrialTimePointDescription"))
                  for h in known if h.get("ClinicalTrialTimePointID")}
    visits = {(s["header"].get("ClinicalTrialTimePointID"), s["header"].get("ClinicalTrialTimePointDescription"))
              for s in series if is_original(s) and s["header"].get("ClinicalTrialTimePointID")}
    visit_source = "original_MR"
    if not visits:
        visits = {(h.get("ClinicalTrialTimePointID"), h.get("ClinicalTrialTimePointDescription"))
                  for h in known if h.get("Modality") == "MR" and str(h.get("ClinicalTrialTimePointID", "")).startswith("T")}
        visit_source = "derived_MR_only"
    record["original_visit"] = next(iter(visits))[0] if len(visits) == 1 else None
    record["original_visit_description"] = next(iter(visits))[1] if len(visits) == 1 else None
    record["visit_assignment_source"] = visit_source
    record["normalized_visit"] = {"T1": "T0", "T2": "T1", "T3": "T2", "T4": "T3"}.get(record["original_visit"])
    record["visit_header_values"] = [list(v) for v in sorted(all_visits)]
    if len(visits) != 1:
        record["qc"]["reasons"].append("missing_or_inconsistent_visit_tags")
    candidates = [s for s in series if excluded_reason(s) is None]
    record["original_candidates"] = [{"series_uid": s["metadata"]["SeriesInstanceUID"],
                                       "description": s["header"].get("SeriesDescription"),
                                       "series_number": s["header"].get("SeriesNumber"),
                                       "nfiles": s["nfiles"]} for s in candidates]
    derived = [s for s in series if s.get("header", {}).get("ucsf", {}).get("total_phases") is not None
               and s["header"].get("Modality") == "MR"]
    evidence = []
    anchors = []
    for d in derived:
        dh = d["header"]
        sn = number(dh.get("SeriesNumber"), 0)
        sref = sn // 10000
        possible = []
        for s in candidates:
            h = s["header"]
            original_sn = number(h.get("SeriesNumber"), -1)
            ref = original_sn // 100 if original_sn >= 100 and "PHILIPS" in str(h.get("Manufacturer", "")).upper() else original_sn
            if ref == sref:
                possible.append(s)
        if len(possible) == 1:
            anchors.append(possible[0])
            evidence.append({"derived_series_uid": d["metadata"]["SeriesInstanceUID"],
                             "derived_series_number": sn, "original_sref": sref,
                             "ucsf": dh["ucsf"], "source_header": dh["file"]})
    unique_anchors = {s["metadata"]["SeriesInstanceUID"]: s for s in anchors}
    record["phase_evidence"] = evidence
    fallback = False
    if len(unique_anchors) != 1:
        anchor = explicit_prepost_anchor(candidates) if not unique_anchors else None
        if anchor is None:
            record["qc"]["reasons"].append("no_unique_UCSF_SER_or_explicit_pre_post_anchor")
            return record
        fallback = True
        total = None
        record["phase_evidence"] = [{"type": "explicit_pre_post_description_and_clock_order",
                                     "precontrast_series_uid": anchor["metadata"]["SeriesInstanceUID"],
                                     "precontrast_description": anchor["header"].get("SeriesDescription"),
                                     "source_header": anchor["header"]["file"]}]
    else:
        anchor = next(iter(unique_anchors.values()))
        totals = {number(e["ucsf"]["total_phases"]) for e in evidence}
        indices = {tuple(e["ucsf"]["ser_phase_indices"] or []) for e in evidence}
        if len(totals) != 1 or len(indices) != 1:
            record["qc"]["reasons"].append("inconsistent_UCSF_phase_evidence")
            return record
        total = next(iter(totals))
        ser_indices = next(iter(indices))
        if total is None or total < 2 or len(ser_indices) != 3 or number(ser_indices[0]) != 0:
            record["qc"]["reasons"].append("precontrast_phase_zero_not_established")
            return record
    family = related_series(anchor, candidates, allow_sequence_match=not fallback)
    # A complete multi-temporal anchored series is already the official DCE;
    # similarly named later scans (e.g. post-needle) are not extra phases.
    family.sort(key=lambda s: s["metadata"]["SeriesInstanceUID"] != anchor["metadata"]["SeriesInstanceUID"])
    allphases = []
    try:
        for s in family:
            uid = s["metadata"]["SeriesInstanceUID"]
            cache = cache_dir / (uid + ".json")
            if cache.exists():
                headers = json.loads(cache.read_text())
            else:
                headers = [read_header(f) for f in dcm_files(root, s["metadata"])]
                write_json(cache, headers)
            groups, method = split_phases(headers)
            for local_index, group in enumerate(groups):
                times = [clock_seconds(h.get("AcquisitionTime") or h.get("ContentTime")) for h in group]
                times = [x for x in times if x is not None]
                allphases.append({"headers": group, "series_uid": uid,
                                  "series_number": s["header"].get("SeriesNumber"),
                                  "local_phase_index": local_index,
                                  "temporal_split_method": method,
                                  "time_seconds": float(np.median(times)) if times else None})
            if not fallback and uid == anchor["metadata"]["SeriesInstanceUID"] and len(groups) == total:
                family = [s]
                break
        if len(family) == 1:
            pass  # internal temporal indices already verified against clock times
        elif all(p["time_seconds"] is not None for p in allphases):
            allphases.sort(key=lambda p: (p["time_seconds"], number(p["series_number"], 0), p["local_phase_index"]))
            if len({p["time_seconds"] for p in allphases}) != len(allphases):
                raise ValueError("separate_phase_series_have_ambiguous_equal_times")
        else:
            raise ValueError("separate_phase_series_lack_acquisition_times")
        for observed_index, phase in enumerate(allphases):
            phase["observed_acquisition_index_zero_based"] = observed_index
        if not fallback and len(allphases) > total and len(family) > 1:
            # Some bilateral protocols reuse the same sequence for both
            # breasts. Keep only stacks that physically overlap anchored S0.
            pre_phases = [p for p in allphases if p["series_uid"] == anchor["metadata"]["SeriesInstanceUID"]
                          and p["local_phase_index"] == 0]
            if len(pre_phases) == 1:
                pre_h = pre_phases[0]["headers"]
                iop = np.asarray(pre_h[0]["ImageOrientationPatient"], float)
                norm = np.cross(iop[:3], iop[3:])
                pre_proj = np.asarray([h["ImagePositionPatient"] for h in pre_h], float) @ norm
                pre_min, pre_max = float(pre_proj.min()), float(pre_proj.max())
                retained = []
                ignored = []
                for phase in allphases:
                    proj = np.asarray([h["ImagePositionPatient"] for h in phase["headers"]], float) @ norm
                    phase_min, phase_max = float(proj.min()), float(proj.max())
                    overlap = max(0., min(pre_max, phase_max)-max(pre_min, phase_min))
                    denominator = min(pre_max-pre_min, phase_max-phase_min)
                    if denominator > 0 and overlap/denominator >= .5:
                        retained.append(phase)
                    else:
                        ignored.append({k: v for k, v in phase.items() if k != "headers"})
                if ignored:
                    record["ignored_nonoverlapping_stacks"] = ignored
                    allphases = retained
        if not fallback and len(allphases) > total and len(family) > 1:
            # Some Siemens exams contain repeated pre-injection acquisitions
            # between S0 and the true first post. The private assumed injection
            # clock identifies the official first-post start. Never infer the
            # second stored series in this situation.
            injection_times = {clock_seconds(e["ucsf"].get("injection_time")) for e in evidence}
            injection_times.discard(None)
            if len(injection_times) == 1:
                injection = next(iter(injection_times))
                after_injection = [p for p in allphases if p["time_seconds"] is not None
                                   and p["time_seconds"] >= injection-10]
                pre = [p for p in allphases if p["series_uid"] == anchor["metadata"]["SeriesInstanceUID"]
                       and p["local_phase_index"] == 0]
                if (len(pre) == 1 and pre[0]["time_seconds"] < injection-10
                        and len(after_injection) == total-1
                        and abs(after_injection[0]["time_seconds"]-injection) <= 10):
                    included = {id(p) for p in pre+after_injection}
                    record["ignored_extra_preinjection_acquisitions"] = [
                        {k: v for k, v in p.items() if k != "headers"}
                        for p in allphases if id(p) not in included]
                    allphases = pre+after_injection
        record["phase_count_observed"] = len(allphases)
        record["phase_count_expected"] = total
        record["phase_summary"] = [{k: v for k, v in p.items() if k != "headers"} for p in allphases]
        if total is not None and len(allphases) != total:
            raise ValueError("observed_phase_count_does_not_match_UCSF_total")
        if len(allphases) < 2:
            raise ValueError("fewer_than_two_phases")
        if allphases[0]["series_uid"] != anchor["metadata"]["SeriesInstanceUID"] or allphases[0]["local_phase_index"] != 0:
            raise ValueError("anchored_precontrast_is_not_first_acquisition")
        ref_ordered, ref_geo, ref_reasons = volume_geometry(allphases[0]["headers"])
        if ref_reasons:
            raise ValueError("invalid_phase0_geometry:" + ",".join(ref_reasons))
        phase_grid_changed = False
        for phase in allphases[1:]:
            phase_ordered, phase_geo, phase_reasons = volume_geometry(phase["headers"])
            if phase_reasons:
                raise ValueError("invalid_phase_geometry:" + ",".join(phase_reasons))
            if (len(phase_ordered) != len(ref_ordered)
                    or not np.allclose(phase_geo["positions"], ref_geo["positions"], atol=.05)
                    or not np.allclose(phase_geo["pixel_spacing"], ref_geo["pixel_spacing"], atol=1e-4)):
                phase_grid_changed = True
        if phase_grid_changed:
            record["qc"]["warnings"] = ["phase_grids_differ; independent_native_firstpost_inference_only"]
        selected = allphases[1]
        if fallback and not re.search(r"\bPOST\b", str(selected["headers"][0].get("SeriesDescription") or ""), re.I):
            raise ValueError("explicit_postcontrast_is_not_second_acquisition")
        if fallback:
            family_uids = {s["metadata"]["SeriesInstanceUID"] for s in family}
            for other in candidates:
                h = other["header"]
                if other["metadata"]["SeriesInstanceUID"] in family_uids:
                    continue
                text = str(h.get("SeriesDescription") or "")
                t = clock_seconds(h.get("AcquisitionTime") or h.get("ContentTime"))
                if (t is not None and t < selected["time_seconds"]
                        and re.search(r"\bPOST\b|\+C\b|\bPOS\b|DYNAMIC|DCE|PRE/POST", text, re.I)):
                    raise ValueError("whole_study_contains_earlier_post_or_dynamic_acquisition")
        ordered, geo, reasons = volume_geometry(selected["headers"])
        record["qc"]["reasons"].extend(reasons)
        record["first_post"] = {"files": [h["file"] for h in ordered],
                                "sop_instance_uids": [h["SOPInstanceUID"] for h in ordered],
                                "series_uid": selected["series_uid"],
                                "phase_index_zero_based": 1,
                                "semantic_phase": "first_postcontrast",
                                "phase_index_semantics": "legacy contrast state: pre=0, firstpost=1; not observed acquisition order",
                                "observed_acquisition_index_zero_based": selected["observed_acquisition_index_zero_based"],
                                "observed_acquisition_index_scope": "chronological original DCE family before extra-pre or contralateral selection; not all non-DCE study series",
                                "phase_index_within_series_zero_based": selected["local_phase_index"],
                                "selection_evidence": ("explicit_pre_post_description_and_clock_order" if fallback else "UCSF_SER_S0_anchor_and_total_phases; first_complete_acquisition_after_phase0"),
                                "temporal_split_method": selected["temporal_split_method"],
                                "geometry": geo,
                                "image_type": ordered[0].get("ImageType"),
                                "series_description": ordered[0].get("SeriesDescription"),
                                "source_header": ordered[0]}
        record["first_post"]["selection_details"] = {
            "official_ucsf_phase_evidence": [e["ucsf"] for e in evidence],
            "ignored_extra_preinjection_acquisitions": record.get("ignored_extra_preinjection_acquisitions", []),
            "ignored_nonoverlapping_stacks": record.get("ignored_nonoverlapping_stacks", []),
            "qc_warnings": record["qc"].get("warnings", [])}
        record["first_post"]["input_scope"] = (
            "complete unilateral stack matching official S0; contralateral separately acquired volumes omitted"
            if record.get("ignored_nonoverlapping_stacks") else
            "complete native first-postcontrast volume for this study")
        if not record["qc"]["reasons"]:
            record["status"] = "ready"
    except Exception as exc:
        record["qc"]["reasons"].append(str(exc))
    record["qc"]["reasons"] = sorted(set(record["qc"]["reasons"]))
    return record


def freeze_manifest(report, reps, output):
    """Freeze sources once, enrich descriptive QC and produce cohort coverage."""
    output = Path(output)
    frozen_path = output / "dce_manifest_v1.json"
    if frozen_path.exists():
        raise FileExistsError(f"Frozen manifest already exists: {frozen_path}; use a new version/output directory")
    studies = defaultdict(list)
    for s in reps:
        studies[s["metadata"]["StudyInstanceUID"]].append(s)
    categories = Counter()
    per_visit = defaultdict(Counter)
    review_rows = []
    for c in report["cases"]:
        ss = studies[c["study_uid"]]
        if c.get("original_visit") is None:
            visits = {(s["header"].get("ClinicalTrialTimePointID"), s["header"].get("ClinicalTrialTimePointDescription"))
                      for s in ss if s["header"].get("Modality") == "MR"
                      and str(s["header"].get("ClinicalTrialTimePointID", "")).startswith("T")}
            if len(visits) == 1:
                c["original_visit"], c["original_visit_description"] = next(iter(visits))
                c["normalized_visit"] = {"T1": "T0", "T2": "T1", "T3": "T2", "T4": "T3"}.get(c["original_visit"])
                c["visit_assignment_source"] = "derived_MR_only"
                c["qc"]["reasons"] = [r for r in c["qc"]["reasons"] if r != "missing_or_inconsistent_visit_tags"]
        else:
            c.setdefault("visit_assignment_source", "original_MR")
        fp = c.get("first_post")
        if fp:
            fp["selection_details"] = {
                "official_ucsf_phase_evidence": [e["ucsf"] for e in c.get("phase_evidence", []) if "ucsf" in e],
                "ignored_extra_preinjection_acquisitions": c.get("ignored_extra_preinjection_acquisitions", []),
                "ignored_nonoverlapping_stacks": c.get("ignored_nonoverlapping_stacks", []),
                "qc_warnings": c["qc"].get("warnings", [])}
            fp["input_scope"] = ("complete unilateral stack matching official S0; contralateral separately acquired volumes omitted"
                                 if c.get("ignored_nonoverlapping_stacks") else
                                 "complete native first-postcontrast volume for this study")
        per_visit[c.get("normalized_visit") or "unknown"][c["status"]] += 1
        if c["status"] == "ready":
            continue
        raw = [s for s in ss if is_original(s)]
        dce_raw = [s for s in raw if excluded_reason(s) is None]
        descriptions = [str(s["header"].get("SeriesDescription") or "") for s in dce_raw]
        processed_dce = [s for s in ss if not is_original(s) and re.search(r"DCE (PRE|POST)", str(s["header"].get("SeriesDescription") or ""), re.I)]
        if c["qc"]["reasons"] != ["no_unique_UCSF_SER_or_explicit_pre_post_anchor"]:
            category = "temporal_or_geometry_qc_requires_review"
        elif processed_dce:
            category = "processed_bilateral_DCE_without_verified_original_anchor"
        elif not dce_raw:
            category = "no_eligible_original_DCE_in_study"
        elif descriptions and all(re.search(r"\bPOST\b", d, re.I) and not re.search(r"\bPRE\b", d, re.I) for d in descriptions):
            category = "all_candidate_volumes_post_labeled_without_verified_pre"
        elif any(re.search(r"DYNAMIC|DYN|FL3D|SPGR|FGRE|DCE", d, re.I) for d in descriptions):
            category = "dynamic_sources_without_unique_verified_phase_anchor"
        else:
            category = "original_sources_without_unique_verified_pre_post_family"
        c["review_category"] = category
        categories[category] += 1
        review_rows.append({"case_id": c["case_id"], "patient_id": c["patient_id"], "study_uid": c["study_uid"],
                            "normalized_visit": c["normalized_visit"], "category": category,
                            "reasons": ";".join(c["qc"]["reasons"]),
                            "original_candidate_descriptions": " | ".join(descriptions),
                            "processed_DCE_descriptions": " | ".join(str(s["header"].get("SeriesDescription")) for s in processed_dce)})
    ready_patients = {c["patient_id"] for c in report["cases"] if c["status"] == "ready"}
    all_patients = {c["patient_id"] for c in report["cases"]}
    report["manifest_version"] = 1
    report["inventory_complete"] = True
    report["frozen"] = True
    report["summary"].update({"patients_with_ready_studies": len(ready_patients),
                              "patients_without_ready_studies": len(all_patients-ready_patients),
                              "patients_with_review_studies": len({r["patient_id"] for r in review_rows}),
                              "patients_without_ready_study_ids": sorted(all_patients-ready_patients),
                              "by_normalized_visit": {k: dict(v) for k, v in sorted(per_visit.items())},
                              "review_categories": dict(categories),
                              "review_reasons": dict(Counter(r for c in report["cases"] for r in c["qc"]["reasons"]))})
    pilot_path = output / "pilot_diverse_manifest.json"
    comparisons = []
    if pilot_path.exists():
        final_cases = {c["case_id"]: c for c in report["cases"]}
        for pilot in json.loads(pilot_path.read_text())["cases"]:
            current = final_cases[pilot["case_id"]]
            pfp, cfp = pilot.get("first_post") or {}, current.get("first_post") or {}
            comparisons.append({"case_id": pilot["case_id"], "status": current["status"],
                                "files_identical": pfp.get("files") == cfp.get("files"),
                                "series_identical": pfp.get("series_uid") == cfp.get("series_uid"),
                                "geometry_identical": pfp.get("geometry") == cfp.get("geometry")})
        if not all(c["files_identical"] and c["series_identical"] and c["geometry_identical"] and c["status"] == "ready" for c in comparisons):
            raise RuntimeError("Pilot source differs from final manifest; inspect before freezing")
    write_json(output / "pilot_to_final_source_comparison.json", {"cases": comparisons, "all_sources_identical": bool(comparisons)})
    with (output / "review_categories.csv").open("w", newline="") as f:
        fields = ["case_id", "patient_id", "study_uid", "normalized_visit", "category", "reasons",
                  "original_candidate_descriptions", "processed_DCE_descriptions"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(review_rows)
    write_json(output / "summary_v1.json", report["summary"])
    write_json(frozen_path, report)
    return frozen_path


RECOVERY_FIELDS = [
    "SOPInstanceUID", "SeriesInstanceUID", "StudyInstanceUID", "SeriesDescription", "ImageType",
    "SeriesNumber", "InstanceNumber", "AcquisitionNumber", "TemporalPositionIdentifier",
    "NumberOfTemporalPositions", "TemporalResolution", "AcquisitionDate", "AcquisitionDateTime",
    "AcquisitionTime", "ContentTime", "TriggerTime", "FrameReferenceTime", "ImageComments",
    "ProtocolName", "SequenceName", "ScanningSequence", "MRAcquisitionType", "RepetitionTime",
    "EchoTime", "InversionTime", "FlipAngle", "Rows", "Columns", "PixelSpacing",
    "ImageOrientationPatient", "ImagePositionPatient", "SliceThickness", "SpacingBetweenSlices",
    "ContrastBolusAgent", "ContrastBolusRoute", "ContrastBolusVolume", "ContrastBolusStartTime",
    "ContrastBolusStopTime", "ContrastBolusTotalDose", "ContrastFlowRate", "ContrastFlowDuration",
    "ContrastBolusIngredient", "ContrastBolusIngredientConcentration", "DerivationDescription",
    "NumberOfAverages", "SeriesTime", "PerformedProcedureStepStartTime"]


def recovery_private(ds):
    selected = {}
    for elem in ds:
        if elem.tag.is_private and elem.VR != "SQ" and re.search(
                r"phase|dynamic|contrast|bolus|duration|acquisition.?time|scan.?time|trigger", elem.name, re.I):
            if isinstance(elem.value, bytes) and len(elem.value) > 4096:
                continue
            selected[str(elem.tag)] = {"name": elem.name, "value": js(elem.value)}
    csa = {}
    try:
        from nibabel.nicom.csareader import read as read_csa
        for tag in [(0x0029, 0x1010), (0x0029, 0x1020)]:
            if tag not in ds or not isinstance(ds[tag].value, bytes):
                continue
            parsed = read_csa(ds[tag].value)
            for name, item in parsed.get("tags", {}).items():
                if name == "MrPhoenixProtocol":
                    lines = []
                    for text in item.get("items", []):
                        if isinstance(text, bytes):
                            text = text.decode("latin-1", errors="replace")
                        lines.extend(line.strip() for line in str(text).splitlines()
                                     if re.search(r"contrast|bolus|inject|lRepetitions|lContrasts|ScanTime|DelayTime", line, re.I))
                    csa[name] = lines
                elif re.search(r"phase|dynamic|contrast|duration|acquisition.?time|scan.?time|trigger|repetition", name, re.I):
                    csa[name] = js(item.get("items"))
    except Exception as exc:
        csa["parse_note"] = type(exc).__name__ + ":" + str(exc)
    return {"known_private_timing": selected, "siemens_csa_timing": csa,
            "ucsf": {name: private(ds, tag) for name, tag in [
                ("total_phases", 0x1030), ("phase_duration", 0x1031), ("injection_time", 0x1033),
                ("effective_delays", 0x1034), ("ser_phase_indices", 0x1035),
                ("timing_method", 0x103A), ("timing_comments", 0x103B)]}}


def aggregate_values(headers):
    result = {}
    for key in RECOVERY_FIELDS:
        values = Counter(json.dumps(h.get(key), sort_keys=True) for h in headers)
        if len(values) == 1 and next(iter(values)) == "null":
            continue
        # Preserve exact textual comments/bolus tags and temporal values.
        if key in ["SOPInstanceUID", "ImagePositionPatient", "InstanceNumber"]:
            result[key] = {"distinct_values": len(values)}
        else:
            result[key] = [{"value": json.loads(value), "count": count} for value, count in values.most_common()]
    return result


def recovery_series(task):
    root, s, cache_dir = task
    uid = s["metadata"]["SeriesInstanceUID"]
    path = cache_dir / (uid + ".json")
    if path.exists():
        return json.loads(path.read_text())
    files = dcm_files(root, s["metadata"])
    ds = pydicom.dcmread(files[0], stop_before_pixels=True)
    result = {"patient_id": s["metadata"]["PatientID"], "study_uid": s["metadata"]["StudyInstanceUID"],
              "series_uid": uid, "modality": str(ds.Modality), "description": str(ds.get("SeriesDescription", "")),
              "representative_file": files[0], "private": recovery_private(ds), "nfiles": len(files),
              "representative": {k: js(ds.get(k)) for k in RECOVERY_FIELDS}}
    # Exhaustively read relevant original magnitude candidates. Other series
    # remain in the whole-study audit with representative evidence.
    if excluded_reason(s) is None:
        headers = []
        for file in files:
            d = pydicom.dcmread(file, stop_before_pixels=True, specific_tags=RECOVERY_FIELDS)
            h = {k: js(d.get(k)) for k in RECOVERY_FIELDS}
            h["file"] = file
            headers.append(h)
        result["headers"] = headers
        result["aggregate"] = aggregate_values(headers)
        try:
            phases, method = split_phases(headers)
            result["temporal_split_method"] = method
            result["phases"] = []
            for idx, phase in enumerate(phases):
                ordered, geometry, reasons = volume_geometry(phase)
                result["phases"].append({"phase_index_within_series_zero_based": idx,
                                         "nfiles": len(phase), "geometry": geometry,
                                         "geometry_reasons": reasons, "aggregate": aggregate_values(phase),
                                         "files": [h["file"] for h in ordered]})
        except Exception as exc:
            result["temporal_split_error"] = str(exc)
    write_json(path, result)
    return result


def recovery_audit(args):
    frozen = json.loads((args.output / "dce_manifest_v1.json").read_text())
    reps = json.loads((args.output / "series_representatives.json").read_text())
    review_ids = {c["study_uid"] for c in frozen["cases"] if c["status"] == "review"}
    # Include all fallback and extra-pre cases in semantic audit because SER
    # first-post assumptions must also agree with the entire study.
    additional = {c["study_uid"] for c in frozen["cases"] if c["status"] == "ready" and (
        c.get("ignored_extra_preinjection_acquisitions") or
        (c.get("first_post") or {}).get("selection_evidence") == "explicit_pre_post_description_and_clock_order")}
    selected = [s for s in reps if s["metadata"]["StudyInstanceUID"] in review_ids | additional]
    dest = args.output / "recovery_v2"
    cache = dest / "extended_header_cache"
    cache.mkdir(parents=True, exist_ok=True)
    results = []
    with ThreadPoolExecutor(args.workers) as ex:
        for idx, item in enumerate(ex.map(recovery_series, ((args.root, s, cache) for s in selected)), 1):
            results.append(item)
            if idx % 50 == 0:
                print(f"recovery series {idx}/{len(selected)}", flush=True)
    summary = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
               "frozen_v1_unchanged": True, "scope": "all 133 reviews plus fallback and ignored-extra-pre semantic audit",
               "review_studies": len(review_ids), "additional_semantic_studies": len(additional),
               "series": len(results), "fully_read_series": sum("headers" in r for r in results),
               "fully_read_instances": sum(len(r.get("headers", [])) for r in results),
               "candidate_fields": RECOVERY_FIELDS,
               "series_cache_directory": str(cache),
               "series_summary": [{k: v for k, v in item.items() if k not in ["headers", "phases"]} for item in results]}
    write_json(args.output / "recovery_audit_v2_headers.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ["series_summary", "candidate_fields"]}, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/public_fast_storage/cxx26/ISPY1"))
    ap.add_argument("--output", type=Path, default=Path("/public_fast_storage/cxx26/ISPY1/segmentation_mamamia_20261026/inventory"))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--recovery-v2", action="store_true", help="Audit expanded contrast and timing fields; never alter v1")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.recovery_v2:
        recovery_audit(args)
        return
    if (args.output / "dce_manifest_v1.json").exists():
        ap.error("Frozen v1 manifest exists; use a new output directory/version instead of overwriting it")
    meta = json.loads((args.root / "metadata/series.json").read_text())
    cache = args.output / "series_representatives.json"
    if cache.exists():
        reps = json.loads(cache.read_text())
    else:
        with ThreadPoolExecutor(args.workers) as executor:
            reps = []
            for idx, item in enumerate(executor.map(representative, ((args.root, m) for m in meta)), 1):
                reps.append(item)
                if idx % 500 == 0:
                    print(f"representatives {idx}/{len(meta)}", flush=True)
        write_json(cache, reps)
    studies = defaultdict(list)
    for s in reps:
        studies[(s["metadata"]["PatientID"], s["metadata"]["StudyInstanceUID"])].append(s)
    header_cache = args.output / "phase_header_cache"
    header_cache.mkdir(exist_ok=True)
    cases = []
    with ThreadPoolExecutor(args.workers) as executor:
        tasks = ((args.root, k, ss, header_cache) for k, ss in sorted(studies.items()))
        for idx, case in enumerate(executor.map(scan_study, tasks), 1):
            cases.append(case)
            if idx % 25 == 0:
                print(f"studies {idx}/{len(studies)}; ready={sum(c['status']=='ready' for c in cases)}", flush=True)
                write_json(args.output / "partial_manifest.json", {"schema_version": 1, "inventory_complete": False,
                           "studies_expected": len(studies), "cases": cases})
    counts = Counter(c["status"] for c in cases)
    reasons = Counter(r for c in cases for r in c["qc"]["reasons"])
    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_root": str(args.root), "schema_version": 1,
              "scope": "all_available_patients_and_treatment_visits",
              "clinical_outcome_labels_used": False,
              "selection_rule": "first post contrast, acquisition index1 after UCSF-verified phase0, original native grid",
              "summary": {"patients": len({c['patient_id'] for c in cases}), "studies": len(cases),
                          "status_counts": dict(counts), "review_reasons": dict(reasons)},
              "cases": cases}
    write_json(args.output / "dce_manifest.json", report)
    write_json(args.output / "summary.json", report["summary"])
    freeze_manifest(report, reps, args.output)
    with (args.output / "study_inventory.csv").open("w", newline="") as out:
        fields = ["case_id", "patient_id", "study_uid", "original_visit", "normalized_visit", "status", "n_slices", "series_uid", "reasons"]
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        for c in cases:
            fp = c.get("first_post") or {}
            writer.writerow({**{k: c.get(k) for k in fields[:6]}, "n_slices": len(fp.get("files", [])),
                             "series_uid": fp.get("series_uid"), "reasons": ";".join(c["qc"]["reasons"])})
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
