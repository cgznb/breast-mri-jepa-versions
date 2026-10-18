#!/usr/bin/env bash

set -euo pipefail

root=/data1/cxx/ispy2_acrin_t0_registration_v1
code_root="$root/code/registration"
python=/data/cxx/ispy2_multiserver/registration-repo/.venv/bin/python
log_dir="$root/logs"
log_path="$log_dir/full_pipeline.log"
state_path="$root/manifest/full_pipeline_state.json"
lock_path="$root/.full_pipeline.lock"

mkdir -p "$log_dir" "$root/manifest"
exec 9>"$lock_path"
if ! flock -n 9; then
    echo "Another Qingyuan full pipeline process holds $lock_path" >&2
    exit 2
fi

exec >>"$log_path" 2>&1
cd "$code_root"
export PYTHONPATH=src

write_state() {
    local stage=$1
    local status=$2
    "$python" - "$state_path" "$stage" "$status" <<'PY'
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

destination = Path(sys.argv[1])
payload = {
    "stage": sys.argv[2],
    "status": sys.argv[3],
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
descriptor, temporary_name = tempfile.mkstemp(
    prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_name, destination)
except Exception:
    Path(temporary_name).unlink(missing_ok=True)
    raise
PY
}

run_allow_exclusions() {
    local stage=$1
    shift
    write_state "$stage" running
    echo "[$(date -Is)] starting $stage"
    set +e
    "$@"
    local status=$?
    set -e
    echo "[$(date -Is)] $stage exit status: $status"
    if (( status > 1 )); then
        write_state "$stage" failed
        exit "$status"
    fi
    write_state "$stage" completed_with_status_$status
}

run_required() {
    local stage=$1
    shift
    write_state "$stage" running
    echo "[$(date -Is)] starting $stage"
    set +e
    "$@"
    local status=$?
    set -e
    echo "[$(date -Is)] $stage exit status: $status"
    if (( status != 0 )); then
        write_state "$stage" failed
        exit "$status"
    fi
    write_state "$stage" completed
}

preparation_summary_complete() {
    local summary_path=$1
    local expected_visits=$2
    "$python" - "$summary_path" "$expected_visits" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = int(sys.argv[2])
if not path.is_file():
    raise SystemExit(1)
payload = json.loads(path.read_text(encoding="utf-8"))
complete = (
    int(payload.get("planned_visits", -1)) == expected
    and int(payload.get("prepared_visits", -1))
    + int(payload.get("failed_items", -1))
    == expected
)
raise SystemExit(0 if complete else 1)
PY
}

write_state waiting_for_acrin_deep_audit running
while pgrep -f 'ispy2_registration[.]cli audit-dicom .*--collection ACRIN-6698' >/dev/null; do
    echo "[$(date -Is)] waiting for ACRIN deep audit"
    sleep 60
done

audit_path="$root/manifest/acrin_6698_dicom_audit.json"
if [[ ! -f "$audit_path" ]] || [[ $("$python" -c 'import json,sys; print(str(bool(json.load(open(sys.argv[1]))["deep"])).lower())' "$audit_path") != true ]]; then
    write_state waiting_for_acrin_deep_audit failed
    echo "ACRIN deep audit did not publish a deep=true result" >&2
    exit 2
fi
write_state waiting_for_acrin_deep_audit completed

if preparation_summary_complete \
    "$root/preprocessed/manifest/ispy2_preparation_summary.json" 221; then
    write_state prepare_ispy2 already_completed
    echo "[$(date -Is)] reusing completed ISPY2 preparation"
else
    run_allow_exclusions prepare_ispy2 \
        "$python" -m ispy2_registration.cli prepare-dicom \
        --dicom-root \
            /data1/cxx/manifest-1782893659073 \
            "$root/retry_dicom/ispy2_part4" \
        --output-root "$root/preprocessed" \
        --collection ISPY2 \
        --workers 4 \
        --resume \
        --skip-dwi
fi

run_allow_exclusions prepare_acrin \
    "$python" -m ispy2_registration.cli prepare-dicom \
    --dicom-root \
        /data1/cxx/ACRIN-6698-I-SPY2-Breast-DWI-manifest \
        "$root/retry_dicom/acrin" \
    --output-root "$root/preprocessed" \
    --collection ACRIN-6698 \
    --workers 4 \
    --resume \
    --skip-dwi

write_state combine_manifests running
"$python" -m ispy2_registration.cli combine-manifests \
    --manifests \
        "$root/preprocessed/manifest/acrin_6698_preprocessed_manifest.csv" \
        "$root/preprocessed/manifest/ispy2_preprocessed_manifest.csv" \
    --destination "$root/manifest/ispy2_acrin_preprocessed_manifest.csv"
"$python" -m ispy2_registration.cli audit \
    --config config/qingyuan_ispy2_acrin_registration.toml \
    --manifest "$root/manifest/ispy2_acrin_preprocessed_manifest.csv"
write_state combine_manifests completed

run_allow_exclusions register_cohort \
    "$python" -m ispy2_registration.cli register-cohort \
    --config config/qingyuan_ispy2_acrin_registration.toml \
    --manifest "$root/manifest/ispy2_acrin_preprocessed_manifest.csv" \
    --output-root "$root/registered" \
    --workers 8 \
    --elastix-threads 3 \
    --resume \
    --skip-dwi

run_required generate_final_report \
    "$python" -m ispy2_registration.cli generate-final-report \
    --registered-manifest \
        "$root/registered/manifest/ispy2_breastdcedl_registered_to_T0_manifest.csv" \
    --uid-status \
        "$root/manifest/acrin_uid_status_final.csv" \
        "$root/manifest/ispy2_part4_uid_status_final.csv" \
    --failures \
        "$root/manifest/acrin_6698_preparation_exclusions.csv" \
        "$root/manifest/ispy2_preparation_exclusions.csv" \
        "$root/preprocessed/manifest/acrin_6698_preparation_failures.csv" \
        "$root/preprocessed/manifest/ispy2_preparation_failures.csv" \
        "$root/registered/manifest/registration_failures.json" \
    --output-root "$root"

write_state dce_first_complete completed
echo "[$(date -Is)] Qingyuan DCE-first pipeline complete; DWI remains deferred"
