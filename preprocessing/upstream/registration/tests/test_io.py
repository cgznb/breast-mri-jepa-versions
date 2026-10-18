from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from ispy2_registration.io import (
    atomic_output_directory,
    output_image_path,
    pair_is_complete,
    read_nifti,
    write_json,
    write_nifti_identity,
)


def test_nifti_round_trip_keeps_zyx_array_and_identity_affine(tmp_path: Path) -> None:
    array = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    path = tmp_path / "image.nii.gz"

    write_nifti_identity(path, array)

    loaded = read_nifti(path, dtype=np.float32)
    image = nib.load(path)
    assert np.array_equal(loaded, array)
    assert np.array_equal(image.affine, np.eye(4))
    assert image.get_data_dtype() == np.dtype(np.float32)


def test_output_image_path_preserves_modality_and_basename(tmp_path: Path) -> None:
    source = Path("/input/ISPY2-1/T1/dce/phase_0.nii.gz")

    result = output_image_path(tmp_path, "ISPY2-1", "T1", "dce", source)

    assert result == tmp_path / "ISPY2-1/T1/dce/phase_0.nii.gz"


def test_atomic_output_directory_publishes_only_after_success(tmp_path: Path) -> None:
    final = tmp_path / "patient/T1"

    with atomic_output_directory(final) as temporary:
        assert not final.exists()
        (temporary / "file.txt").write_text("complete", encoding="utf-8")

    assert (final / "file.txt").read_text(encoding="utf-8") == "complete"
    assert not list(final.parent.glob(".T1.tmp-*"))


def test_atomic_output_directory_removes_temporary_on_failure(tmp_path: Path) -> None:
    final = tmp_path / "patient/T1"

    with pytest.raises(RuntimeError, match="stop"):
        with atomic_output_directory(final) as temporary:
            (temporary / "partial.txt").write_text("partial", encoding="utf-8")
            raise RuntimeError("stop")

    assert not final.exists()
    assert not list(final.parent.glob(".T1.tmp-*"))


def test_pair_is_complete_requires_terminal_status_and_all_declared_files(
    tmp_path: Path,
) -> None:
    pair_dir = tmp_path / "T1"
    pair_dir.mkdir()
    output = pair_dir / "dce/phase.nii.gz"
    write_json(
        pair_dir / "registration.json",
        {"registration_status": "deformable", "output_files": [str(output)]},
    )

    assert not pair_is_complete(pair_dir)
    write_nifti_identity(output, np.zeros((1, 2, 3), dtype=np.float32))
    assert pair_is_complete(pair_dir)

    payload = json.loads((pair_dir / "registration.json").read_text(encoding="utf-8"))
    payload["registration_status"] = "failed"
    write_json(pair_dir / "registration.json", payload)
    assert not pair_is_complete(pair_dir)
