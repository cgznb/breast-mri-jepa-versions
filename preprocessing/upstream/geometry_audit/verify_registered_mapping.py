"""Replay same-phase MRI samples through the retained T0 coordinate mapping."""

import csv
import itertools
import json
import sys

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from nibabel.spaces import vox2out_vox
from scipy.ndimage import map_coordinates

import audit
from registered_audit import affine_apply, native_affine

sys.path.insert(0, str(audit.ROOT / "MAM/longitudinal_temporal_pillar"))
from src.generated_dce0 import native_geometry_from_meta


def main():
    inventory = audit.read(audit.SEG / "inventory.json")
    seg_index = {(r["patient_id"],r["visit"]):r for r in inventory["records"]}
    with (audit.BUNDLE / "visits.csv").open(newline="") as handle:
        visits = {r["patient_id"]:r for r in csv.DictReader(handle) if r["visit"] == "T0"}
    rows = audit.read(audit.OUT / "registered_new_mask_patient_geometry.json")["new_prediction"]
    eligible = []
    for row in rows:
        record = seg_index[row["native_patient_id"], "T0"]
        saved = audit.Path(visits[row["patient_id"]]["dce0_path"]).with_name(audit.Path(record["source_image"]).name)
        if saved.is_file():
            eligible.append((row,record,saved))
    results = []
    selected = np.unique(np.linspace(0,len(eligible)-1,6,dtype=int))
    for index in selected:
        row,record,saved_path = eligible[index]
        meta = audit.read(audit.Path(visits[row["patient_id"]]["meta_path"]))
        geometry = native_geometry_from_meta(meta, meta["target_geometry"]["shape_zyx"])
        _, strict_affine = vox2out_vox((geometry.shape_zyx[::-1],geometry.affine_ras),voxel_sizes=audit.SPACING[::-1])
        image = sitk.ReadImage(str(audit.SEG / "masks" / (record["case_id"] + ".nii.gz")))
        mask_to_native = np.eye(4)
        mask_to_native[:3,:3] = np.asarray(image.GetDirection()).reshape(3,3) @ np.diag(image.GetSpacing())
        mask_to_native[:3,3] = image.GetOrigin()
        registered_to_native_index = np.linalg.inv(native_affine(record)) @ mask_to_native @ np.linalg.inv(np.asarray(row["mask_to_strict_index_xyz"])) @ np.linalg.inv(strict_affine) @ geometry.affine_ras
        raw = np.asarray(nib.load(record["source_image"]).dataobj,dtype=np.float32)
        saved = np.asarray(nib.load(saved_path).dataobj,dtype=np.float32)
        assert saved.shape == geometry.shape_zyx
        axes = [np.unique(np.linspace(1,size-2,19,dtype=int)) for size in saved.shape[::-1]]
        points = np.asarray(list(itertools.product(*axes)))
        native_points = affine_apply(registered_to_native_index,points)
        inside = np.all((native_points >= 1) & (native_points <= np.asarray(raw.shape[::-1])-2),axis=1)
        points,native_points = points[inside],native_points[inside]
        actual = map_coordinates(raw,native_points[:,::-1].T,order=1,mode="constant",cval=0,prefilter=False)
        expected = saved[tuple(points[:,::-1].T)]
        assert np.count_nonzero(expected) > 100 and np.ptp(expected) > 0
        error = np.abs(actual-expected)
        scale = max(float(np.quantile(np.abs(expected),.99)),1)
        assert float(error.mean()) / scale < 1e-5
        result = {"registered_patient_id":row["patient_id"],"samples":len(points),"nonzero_reference_samples":int(np.count_nonzero(expected)),"reference_sample_range":float(np.ptp(expected)),"same_phase_raw_to_saved_registered_mae":float(error.mean()),"max_absolute_error":float(error.max()),"mae_over_p99":float(error.mean()/scale)}
        results.append(result)
        print(json.dumps(result),flush=True)
    audit.save("independent_registered_image_replay.json",{"available_registered_first_post_cases":len(eligible),"method":"Independent SciPy linear sampling of raw first-post at integer saved registered first-post positions; no intensity normalization", "results":results})


if __name__ == "__main__":
    main()
