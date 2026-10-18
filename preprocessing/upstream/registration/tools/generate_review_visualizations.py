from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


TERMINAL_STATUSES = {"fixed_reference", "deformable", "rigid_fallback"}
FOV_RATIO_THRESHOLD = 1.5
LOW_NON_TUMOR_NCC = 0.5
LOW_PROTECTED_NCC = 0.0
FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
QC_IMAGE_SIZE = (1800, 1350)
BANNER_HEIGHT = 150


@dataclass(frozen=True)
class SourceGeometry:
    shape_zyx: tuple[int, int, int]
    spacing_zyx: tuple[float, float, float]

    @property
    def extent_zyx_mm(self) -> np.ndarray:
        return np.asarray(self.shape_zyx) * np.asarray(self.spacing_zyx)


@dataclass(frozen=True)
class ReviewRecord:
    patient_id: str
    visit: str
    registration_status: str
    qc_path: Path
    source_geometry: SourceGeometry
    fixed_geometry: SourceGeometry
    in_plane_fov_ratio: float
    rigid_non_tumor_ncc: float | None
    rigid_protected_ncc: float | None
    rotation_degrees: float
    translation_mm: float
    rigid_iterations: int | None
    rigid_stop_condition: str
    qc_reasons: tuple[str, ...]

    @property
    def pair_id(self) -> str:
        return f"{self.patient_id}/{self.visit}"

    @property
    def filename_id(self) -> str:
        return f"{self.patient_id}_{self.visit}"

    @property
    def mean_rigid_ncc(self) -> float | None:
        if self.rigid_non_tumor_ncc is None or self.rigid_protected_ncc is None:
            return None
        return (self.rigid_non_tumor_ncc + self.rigid_protected_ncc) / 2.0

    @property
    def reached_maximum_iterations(self) -> bool:
        return "Maximum number of iterations" in self.rigid_stop_condition


@dataclass(frozen=True)
class Thresholds:
    non_tumor_ncc_p05: float
    protected_ncc_p05: float
    rotation_p99_degrees: float
    translation_p99_mm: float

    def as_dict(self) -> dict[str, float]:
        return {
            "non_tumor_ncc_p05": self.non_tumor_ncc_p05,
            "protected_ncc_p05": self.protected_ncc_p05,
            "rotation_p99_degrees": self.rotation_p99_degrees,
            "translation_p99_mm": self.translation_p99_mm,
        }


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def load_source_geometries(manifest_path: Path) -> dict[tuple[str, str], SourceGeometry]:
    geometries: dict[tuple[str, str], SourceGeometry] = {}
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            patient_id = row["patient_id"]
            visit = row["visit"]
            metadata = json.loads(Path(row["meta_path"]).read_text(encoding="utf-8"))
            pixel_spacing = metadata["pixel_spacing"]
            slice_spacing = metadata.get("spacing_between_slices")
            if slice_spacing is None:
                slice_spacing = metadata["slice_thickness"]
            geometries[(patient_id, visit)] = SourceGeometry(
                shape_zyx=(int(row["n_slices"]), int(row["rows"]), int(row["cols"])),
                spacing_zyx=(
                    float(slice_spacing),
                    float(pixel_spacing[0]),
                    float(pixel_spacing[1]),
                ),
            )
    return geometries


def _rigid_motion(transform_path: Path) -> tuple[float, float]:
    text = transform_path.read_text(encoding="utf-8")
    match = re.search(r"^Parameters:\s+(.+)$", text, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"rigid transform lacks parameters: {transform_path}")
    values = tuple(float(value) for value in match.group(1).split())
    if len(values) != 6 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"rigid transform parameters are invalid: {transform_path}")
    rotation = float(np.linalg.norm(values[:3]) * 180.0 / math.pi)
    translation = float(np.linalg.norm(values[3:]))
    return rotation, translation


def _symmetric_ratio(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.maximum(first / second, second / first)


def load_review_records(
    output_root: Path,
    source_geometries: dict[tuple[str, str], SourceGeometry],
) -> tuple[ReviewRecord, ...]:
    records: list[ReviewRecord] = []
    for status_path in sorted(output_root.glob("ISPY2-*/*/registration.json")):
        payload = json.loads(status_path.read_text(encoding="utf-8"))
        visit = str(payload["visit"])
        status = str(payload["registration_status"])
        if visit == "T0" or status not in TERMINAL_STATUSES:
            continue
        patient_id = str(payload["patient_id"])
        fixed_geometry = source_geometries[(patient_id, "T0")]
        source_geometry = source_geometries[(patient_id, visit)]
        extent_ratio = _symmetric_ratio(
            source_geometry.extent_zyx_mm,
            fixed_geometry.extent_zyx_mm,
        )
        qc = payload.get("qc")
        if not isinstance(qc, dict):
            raise ValueError(f"registration status lacks QC: {status_path}")
        rotation, translation = _rigid_motion(
            status_path.parent / "transforms" / "rigid.tfm"
        )
        qc_path = status_path.parent / "qc" / f"{patient_id}_{visit}_to_T0.png"
        if not qc_path.is_file():
            raise FileNotFoundError(f"missing QC image: {qc_path}")
        reasons = qc.get("reasons") or []
        records.append(
            ReviewRecord(
                patient_id=patient_id,
                visit=visit,
                registration_status=status,
                qc_path=qc_path,
                source_geometry=source_geometry,
                fixed_geometry=fixed_geometry,
                in_plane_fov_ratio=float(np.max(extent_ratio[1:])),
                rigid_non_tumor_ncc=_finite_float(qc.get("rigid_non_tumor_ncc")),
                rigid_protected_ncc=_finite_float(qc.get("rigid_protected_ncc")),
                rotation_degrees=rotation,
                translation_mm=translation,
                rigid_iterations=(
                    int(qc["rigid_iterations"])
                    if qc.get("rigid_iterations") is not None
                    else None
                ),
                rigid_stop_condition=str(qc.get("rigid_stop_condition", "")),
                qc_reasons=tuple(str(reason) for reason in reasons),
            )
        )
    return tuple(records)


def cohort_thresholds(records: Sequence[ReviewRecord]) -> Thresholds:
    complete = [
        record
        for record in records
        if record.rigid_non_tumor_ncc is not None
        and record.rigid_protected_ncc is not None
    ]
    if not complete:
        raise ValueError("no records contain both rigid NCC values")
    return Thresholds(
        non_tumor_ncc_p05=float(
            np.percentile([record.rigid_non_tumor_ncc for record in complete], 5.0)
        ),
        protected_ncc_p05=float(
            np.percentile([record.rigid_protected_ncc for record in complete], 5.0)
        ),
        rotation_p99_degrees=float(
            np.percentile([record.rotation_degrees for record in complete], 99.0)
        ),
        translation_p99_mm=float(
            np.percentile([record.translation_mm for record in complete], 99.0)
        ),
    )


def review_rules(record: ReviewRecord, thresholds: Thresholds) -> tuple[str, ...]:
    # The reported three-rule cohort was derived from pairs with both absolute
    # rigid-NCC measurements. B-spline execution failures are reviewed in their
    # separate algorithm-failure group instead of entering through FOV alone.
    if record.rigid_non_tumor_ncc is None or record.rigid_protected_ncc is None:
        return ()
    rules: list[str] = []
    if record.in_plane_fov_ratio >= FOV_RATIO_THRESHOLD:
        rules.append("FOV")
    if (
        record.rigid_non_tumor_ncc is not None
        and record.rigid_protected_ncc is not None
        and record.rigid_non_tumor_ncc < thresholds.non_tumor_ncc_p05
        and record.rigid_protected_ncc < thresholds.protected_ncc_p05
    ):
        rules.append("NCC")
    extreme_motion = (
        record.rotation_degrees >= thresholds.rotation_p99_degrees
        or record.translation_mm >= thresholds.translation_p99_mm
    )
    low_similarity = (
        record.rigid_non_tumor_ncc is not None
        and record.rigid_protected_ncc is not None
        and (
            record.rigid_non_tumor_ncc < LOW_NON_TUMOR_NCC
            or record.rigid_protected_ncc < LOW_PROTECTED_NCC
        )
    )
    if extreme_motion and low_similarity:
        rules.append("MOTION")
    return tuple(rules)


def select_rule_union(
    records: Sequence[ReviewRecord], thresholds: Thresholds
) -> tuple[ReviewRecord, ...]:
    return tuple(record for record in records if review_rules(record, thresholds))


def select_maximum_iteration_examples(
    records: Sequence[ReviewRecord], *, count: int
) -> tuple[ReviewRecord, ...]:
    eligible = [record for record in records if record.reached_maximum_iterations]
    eligible.sort(
        key=lambda record: (
            record.mean_rigid_ncc
            if record.mean_rigid_ncc is not None
            else float("inf"),
            record.patient_id,
            record.visit,
        )
    )
    if len(eligible) < count:
        raise ValueError(
            f"only {len(eligible)} records reached maximum iterations; requested {count}"
        )
    return tuple(eligible[:count])


def _format_number(value: float | None, digits: int = 3) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def _fit_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, width: int) -> str:
    if draw.textlength(text, font=font) <= width:
        return text
    shortened = text
    while shortened and draw.textlength(shortened + "...", font=font) > width:
        shortened = shortened[:-1]
    return shortened.rstrip() + "..."


def annotate_qc_image(
    record: ReviewRecord,
    destination: Path,
    *,
    group_label: str,
    rules: Sequence[str],
) -> None:
    with Image.open(record.qc_path) as source_image:
        source = source_image.convert("RGBA")
    if source.size != QC_IMAGE_SIZE:
        raise ValueError(f"unexpected QC image size: {record.qc_path} {source.size}")
    canvas = Image.new("RGBA", (source.width, source.height + BANNER_HEIGHT), "white")
    canvas.paste(source, (0, BANNER_HEIGHT))
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.truetype(str(FONT_PATH), 30)
    body_font = ImageFont.truetype(str(FONT_PATH), 23)
    small_font = ImageFont.truetype(str(FONT_PATH), 20)
    margin = 28
    usable_width = source.width - 2 * margin
    title = (
        f"{record.pair_id} | {group_label} | selected={record.registration_status}"
    )
    metrics = (
        f"rules={','.join(rules) if rules else 'MAX_ITER'} | "
        f"FOVxy={record.in_plane_fov_ratio:.3f} | "
        f"rigid NCC non-tumor={_format_number(record.rigid_non_tumor_ncc)} | "
        f"protected={_format_number(record.rigid_protected_ncc)} | "
        f"rotation={record.rotation_degrees:.2f} deg | "
        f"translation={record.translation_mm:.2f} mm"
    )
    details = (
        f"rigid iterations={record.rigid_iterations if record.rigid_iterations is not None else 'NA'} | "
        f"candidate QC reasons={','.join(record.qc_reasons) if record.qc_reasons else 'none'}"
    )
    draw.text((margin, 12), _fit_text(draw, title, title_font, usable_width), fill="black", font=title_font)
    draw.text((margin, 57), _fit_text(draw, metrics, body_font, usable_width), fill="black", font=body_font)
    draw.text((margin, 101), _fit_text(draw, details, small_font, usable_width), fill="#333333", font=small_font)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG", optimize=True)


INDEX_FIELDS = (
    "ordinal",
    "patient_id",
    "visit",
    "registration_status",
    "selection_rules",
    "in_plane_fov_ratio",
    "rigid_non_tumor_ncc",
    "rigid_protected_ncc",
    "mean_rigid_ncc",
    "rotation_degrees",
    "translation_mm",
    "rigid_iterations",
    "rigid_stop_condition",
    "qc_reasons",
    "source_qc_path",
    "visualization_path",
)


def _write_group(
    destination: Path,
    records: Sequence[ReviewRecord],
    *,
    group_label: str,
    thresholds: Thresholds,
) -> tuple[Path, ...]:
    files: list[Path] = []
    index_rows: list[dict[str, Any]] = []
    for ordinal, record in enumerate(records, start=1):
        rules = review_rules(record, thresholds) if group_label == "THREE_RULE_UNION" else ()
        path = destination / f"{ordinal:03d}_{record.filename_id}.png"
        annotate_qc_image(record, path, group_label=group_label, rules=rules)
        files.append(path)
        index_rows.append(
            {
                "ordinal": ordinal,
                "patient_id": record.patient_id,
                "visit": record.visit,
                "registration_status": record.registration_status,
                "selection_rules": ";".join(rules) if rules else "MAX_ITER",
                "in_plane_fov_ratio": record.in_plane_fov_ratio,
                "rigid_non_tumor_ncc": record.rigid_non_tumor_ncc,
                "rigid_protected_ncc": record.rigid_protected_ncc,
                "mean_rigid_ncc": record.mean_rigid_ncc,
                "rotation_degrees": record.rotation_degrees,
                "translation_mm": record.translation_mm,
                "rigid_iterations": record.rigid_iterations,
                "rigid_stop_condition": record.rigid_stop_condition,
                "qc_reasons": ";".join(record.qc_reasons),
                "source_qc_path": str(record.qc_path),
                "visualization_path": path.name,
            }
        )
    with (destination / "index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(index_rows)
    return tuple(files)


def _validate_group(destination: Path, *, expected_count: int) -> tuple[Path, ...]:
    files = tuple(sorted(destination.glob("*.png")))
    if len(files) != expected_count:
        raise ValueError(
            f"{destination} contains {len(files)} PNG files; expected {expected_count}"
        )
    expected_size = (QC_IMAGE_SIZE[0], QC_IMAGE_SIZE[1] + BANNER_HEIGHT)
    for path in files:
        with Image.open(path) as image:
            if image.size != expected_size:
                raise ValueError(f"wrong visualization size: {path} {image.size}")
            array = np.asarray(image.convert("RGB"), dtype=np.float32)
        if float(np.std(array)) <= 5.0:
            raise ValueError(f"visualization is blank: {path}")
    with (destination / "index.csv").open(newline="", encoding="utf-8") as handle:
        if sum(1 for _ in csv.DictReader(handle)) != expected_count:
            raise ValueError(f"index row count differs: {destination / 'index.csv'}")
    return files


def _write_readme(path: Path) -> None:
    path.write_text(
        """# Registration Review Visualizations

Each PNG prepends selection metrics to the original registration QC figure.
The three orthogonal rows show T0, rigid, the selected final transform, and the
candidate B-spline Jacobian. No registration artifact was recomputed or changed.

- `three_rule_union_56`: union of FOV, absolute rigid-NCC, and extreme-motion rules.
- `rigid_max_iterations_50`: 50 maximum-iteration pairs ranked by lowest mean rigid NCC.

See each directory's `index.csv` for exact metrics, QC reasons, and source paths.
""",
        encoding="utf-8",
    )


def generate(
    *,
    manifest_path: Path,
    output_root: Path,
    destination: Path,
    max_iteration_count: int,
) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    source_geometries = load_source_geometries(manifest_path)
    records = load_review_records(output_root, source_geometries)
    thresholds = cohort_thresholds(records)
    rule_union = select_rule_union(records, thresholds)
    maximum_iterations = select_maximum_iteration_examples(
        records, count=max_iteration_count
    )
    if len(rule_union) != 56:
        raise ValueError(f"three-rule union changed: expected 56, found {len(rule_union)}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        union_dir = temporary / "three_rule_union_56"
        maximum_dir = temporary / f"rigid_max_iterations_{max_iteration_count}"
        union_dir.mkdir()
        maximum_dir.mkdir()
        _write_group(
            union_dir,
            rule_union,
            group_label="THREE_RULE_UNION",
            thresholds=thresholds,
        )
        _write_group(
            maximum_dir,
            maximum_iterations,
            group_label="RIGID_MAX_ITER",
            thresholds=thresholds,
        )
        _validate_group(union_dir, expected_count=len(rule_union))
        _validate_group(maximum_dir, expected_count=len(maximum_iterations))
        overlap = sorted(
            {record.pair_id for record in rule_union}
            & {record.pair_id for record in maximum_iterations}
        )
        summary = {
            "source_manifest": str(manifest_path),
            "registration_output_root": str(output_root),
            "records_with_terminal_moving_registration": len(records),
            "maximum_iteration_records_in_cohort": sum(
                record.reached_maximum_iterations for record in records
            ),
            "thresholds": thresholds.as_dict(),
            "three_rule_union_count": len(rule_union),
            "maximum_iteration_example_count": len(maximum_iterations),
            "set_overlap_count": len(overlap),
            "set_overlap_pairs": overlap,
            "maximum_iteration_selection": "lowest mean of rigid non-tumor and protected NCC",
        }
        (temporary / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_readme(temporary / "README.md")
        os.replace(temporary, destination)
        return summary
    except Exception:
        import shutil

        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--max-iteration-count", type=int, default=50)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = generate(
        manifest_path=args.manifest,
        output_root=args.output_root,
        destination=args.destination,
        max_iteration_count=args.max_iteration_count,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
