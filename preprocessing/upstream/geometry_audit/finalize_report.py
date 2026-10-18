"""Write a registered-cohort report and T0-only geometry policy comparison."""

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

import audit


def main():
    legacy = audit.read(audit.OUT / "patient_geometry.json")["registered_legacy"]
    mapped = audit.read(audit.OUT / "registered_new_mask_patient_geometry.json")
    legacy_index = {r["patient_id"]: r for r in legacy}
    protected = []
    for row in mapped["new_prediction"]:
        protected.append({
            **row,
            "extent_zyx_mm": np.maximum(row["extent_zyx_mm"],row["legacy_t0_required_zyx_mm"]).tolist(),
            "localization": "new_T0_prediction_center_with_historical_T0_bbox_containment",
        })
    for row in mapped["excluded"]:
        old = legacy_index[row["patient_id"]]
        protected.append({**old,"localization":"empty_new_T0_prediction_fallback_to_existing_registered_T0_mask"})
    train = [r for r in protected if r["fold"] == "train"]
    val = [r for r in protected if r["fold"] == "val"]
    assert len(train) == 765 and len(val) == 102
    assert set(r["patient_id"] for r in protected) == set(legacy_index)
    policy_rows = []
    patient_sizes = []
    collections = {
        "historical_T0": legacy,
        "new_T0_prediction": mapped["new_prediction"],
        "new_T0_center_protected_by_historical_T0": protected,
        "T0_union_recentered_alternative": mapped["t0_union"],
    }
    for name, rows in collections.items():
        for fold in ["train","val"]:
            cohort = [r for r in rows if r["fold"] == fold]
            for margin in [10,15,20,25,30]:
                extent = np.array([r["extent_zyx_mm"] for r in cohort])
                sizes = (np.ceil(((extent+2*margin)/audit.SPACING+1-1e-8)/32)*32).astype(int)
                span = (sizes-1)*audit.SPACING
                assert np.all(span >= extent+2*margin-1e-6)
                required = np.array([r["longitudinal_required_zyx_mm"] for r in cohort])
                full = np.all(required <= span+1e-6,axis=1)
                ratio = sizes.prod(axis=1)/audit.OLD_SHAPE.prod()
                row = {"localization":name,"fold":fold,"patients":len(cohort),"margin_mm":margin,"mean_voxel_ratio":float(ratio.mean()),"median_voxel_ratio":float(np.median(ratio)),"all_historical_followup_bboxes_fit":int(full.sum()),"all_historical_followup_bboxes_fit_percent":float(full.mean()*100),"below_original_voxel_count":int((ratio<1).sum()),"depth32":int((sizes[:,0]==32).sum()),"depth64":int((sizes[:,0]==64).sum()),"depth96":int((sizes[:,0]==96).sum()),"depth128":int((sizes[:,0]==128).sum())}
                policy_rows.append(row)
                if name == "new_T0_center_protected_by_historical_T0":
                    for patient,size,fit in zip(cohort,sizes,full,strict=True):
                        patient_sizes.append({"patient_id":patient["patient_id"],"fold":fold,"margin_mm":margin,"shape_zyx":"x".join(map(str,size)),"historical_followup_bboxes_fit":bool(fit)})
    audit.csv_save("registered_policy_comparison.csv",policy_rows)
    audit.csv_save("proposed_t0_only_patient_sizes.csv",patient_sizes)
    summary = audit.read(audit.OUT / "registered_summary.json")
    summary["registered_patients_requested"] = 867
    summary["registered_patients_with_available_transform"] = 867
    summary["unavailable_transform_count"] = 0
    summary.pop("registered_patients",None)
    audit.save("registered_summary.json",summary)
    fig,axes = plt.subplots(1,2,figsize=(12,4.5),layout="constrained")
    labels = {
        "historical_T0":("Historical registered T0", "#526cbb"),
        "new_T0_prediction":("New T0 prediction only", "#b66030"),
        "new_T0_center_protected_by_historical_T0":("New center + existing T0 extent", "#168178"),
    }
    for name,(label,color) in labels.items():
        data = [r for r in policy_rows if r["localization"]==name and r["fold"]=="train"]
        axes[0].plot([r["mean_voxel_ratio"]*100 for r in data],[r["all_historical_followup_bboxes_fit_percent"] for r in data],"o-",color=color,label=label)
        for r in data:
            if r["margin_mm"] in [20,25,30]:
                axes[0].annotate(str(r["margin_mm"]),(r["mean_voxel_ratio"]*100,r["all_historical_followup_bboxes_fit_percent"]),xytext=(3,4),textcoords="offset points",fontsize=8)
    axes[0].set(xlabel="Mean voxel count (% of 96 x 256 x 256)",ylabel="Patients with all historical visit boxes inside (%)",ylim=(20,102))
    axes[0].legend(fontsize=8,loc="lower right")
    x=np.arange(4)
    for offset,margin,color in [(-.2,20,"#168178"),(.2,25,"#526cbb")]:
        r=next(r for r in policy_rows if r["localization"]=="new_T0_center_protected_by_historical_T0" and r["fold"]=="train" and r["margin_mm"]==margin)
        axes[1].bar(x+offset,[r["depth"+str(d)] for d in [32,64,96,128]],width=.4,color=color,label=f"{margin} mm margin")
    axes[1].set(xticks=x,xticklabels=["32","64","96","128"],xlabel="Required slice count",ylabel="Registered training patients")
    axes[1].legend(fontsize=9)
    for ax in axes:
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(alpha=.2,axis="y")
    fig.savefig(audit.OUT / "registered_crop_tradeoff.png",dpi=180)
    plt.close(fig)
    rows=[r for r in policy_rows if r["localization"]=="new_T0_center_protected_by_historical_T0" and r["fold"]=="train" and r["margin_mm"] in [15,20,25,30]]
    table="\n".join(f"| {r['margin_mm']} | {r['mean_voxel_ratio']*100:.2f}% | {r['all_historical_followup_bboxes_fit']}/765 ({r['all_historical_followup_bboxes_fit_percent']:.2f}%) | {r['depth64']} | {r['depth96']} | {r['depth128']} |" for r in rows)
    modes=Counter(r["shape_zyx"] for r in patient_sizes if r["fold"]=="train" and r["margin_mm"]==25)
    mode_table="\n".join(f"| {shape} | {count} |" for shape,count in modes.most_common(10))
    text=f"""# Smaller T0-Centered Crops for Registered I-SPY2

Date: 2026-09-15. Scope: the historical registered SymmFlow cohort, 765 training
and 102 validation patients. All dimension tuples below are Z/Y/X.

## Recommendation

Use patient-specific rectangular crops at the existing spacing
`2.0 / 0.7032 / 0.7032 mm`. Keep one T0-defined grid for every visit of a patient.
Use 64 slices and roughly 128-192 in-plane voxels as common sizes, while retaining
96/128-slice and wider cases where their T0 anatomy requires them. Rounding each
axis separately to a multiple of 32 saves more voxels than forcing a square box.

For the existing new segmentation model, keep its T0 bbox center, and use the
historical registered T0 mask as an additional minimum-coverage constraint.
Start with 25 mm per-side margin as a conservative geometry candidate; 20 mm
is the smaller alternative. This choice uses only training geometry. It is a
proposal, not an activated preprocessing or training policy.

For each axis, if `E` is twice the largest T0 voxel-face distance from the chosen
center (including the existing T0 mask), `s` is spacing, and `m` is margin:

`N = 32 * ceil((((E + 2*m) / s) + 1) / 32)`

The one training patient with an empty new T0 prediction can retain their existing
nonempty registered T0 mask. No patient is excluded by crop size in this proposal.

## Registered Data Evidence

All 867 registered patients have a corresponding new T0 segmentation file;
866 are nonempty (764 train, 102 val), and one training prediction is empty.
All 866 nonempty masks were read and mapped through their original phase1/phase0
T0 transform into the exact historical Strict-A resampling grid. The 291 remotely
retained transform files were copied locally after transient SSH failures.

New-mask training bbox median Z/Y/X is 35.25/35.89/31.68 mm. The 95th percentiles
are 94.86/102.62/213.81 mm. There are 319/764 multiple-component T0 predictions;
widely separated components contribute to the large in-plane tail. No component
was discarded. These are predicted extents, not adjudicated tumor dimensions.

At fixed spacing, new T0 mask plus 20 mm fits in `64x128x128` for 485/764 (63.48%),
`64x160x160` for 589/764 (77.09%), and `64x192x192` for 629/764 (82.33%). Their
voxel counts are 16.67%, 26.04%, and 37.50% of the original. These rates count
complete T0 mask and margin containment, not follow-up coverage.

New-mask-only adaptive crops plus 20 mm average 24.44% of original voxels, but
only 467/764 (61.13%) fully contain every available historical registered visit
mask box. Differences in mask definition and localization must be addressed
before treating the smallest crop as sufficient for longitudinal generation.

## Center-Preserving T0 Protection

The new T0 center is retained, while crop size must also contain the existing
registered T0 mask. Every crop size is determined from T0 only. Follow-up boxes
are used to compare global margins on the training split, never to choose an
individual patient's crop.

| Margin per side (mm) | Mean original voxel count | All historical visit boxes contained | 64 slices | 96 slices | 128 slices |
|---|---:|---:|---:|---:|---:|
{table}

Most common training sizes with 25 mm margin:

| Z/Y/X | Patients |
|---|---:|
{mode_table}

The 25 mm rule retains all 765 training patients; 665 have fewer voxels than the
original fixed crop. Its 14 incomplete-box cases require review. Bounding-box
containment is conservative: failure does not mean all tumor voxels are lost,
or that the patient must be excluded. No clinical sensitivity or Dice is measured.

A separate T0-only union followed by recentering reduces mean voxels further,
to 46.01% at 25 mm for 764 nonempty-new-mask training patients, with 746/764
historical-visit boxes fully contained. It changes the center and is recorded
as an alternative, not the requested center-preserving default.

## Verification and Limits

- All 3,777 segmentation mask files match their report sizes; report counts and
  extents match the original crop manifest. Twelve independent mask readbacks
  verified counts, bounding boxes and physical geometry.
- All 866 mapped T0 masks passed count/bbox/geometry and physical point roundtrip
  checks; maximum roundtrip error is below `1.5e-13 mm`.
- Five geometry-only Strict-A grids exactly match the existing full-grid helper.
  Six independent SciPy same-phase MRI replays match saved registered image
  samples; see `independent_registered_image_replay.json`.
- Seven meta-tensor encoder/decoder/flow shape checks passed. Current codec/data
  wrappers still enforce `96x256x256`, so this is not a ready-to-launch size change.
- New crop caches and latent statistics are needed. The frozen codec's numerical
  reconstruction quality on tighter crops must be evaluated before flow training;
  architecture shape compatibility alone does not establish that quality.
- Follow-up coverage uses historical registered masks, not mapped new T1-T3
  predictions. This comparison does not establish which mask definition is better.
- Voxel ratios are computational proxies, not measured runtime or GPU savings.
  Validation geometry is reported separately and was not used for the recommendation.

## Artifacts

`registered_policy_comparison.csv`, `proposed_t0_only_patient_sizes.csv`,
`registered_new_mask_candidate_sizes.csv`, `registered_summary.json`,
`registered_crop_tradeoff.png`, and the audit/verification Python scripts.

No existing crop cache, dataset split, model checkpoint, or training process was
changed. The earlier native-cohort files are background inventory only and are
not the basis of this registered-cohort recommendation.
"""
    (audit.OUT / "README.md").write_text(text)
    print(table,flush=True)


if __name__ == "__main__":
    main()
